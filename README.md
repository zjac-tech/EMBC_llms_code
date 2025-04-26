# LLMs for triage

Install related environment

`pip install ms-swift -U`

`pip install transformers -U`


Model deployment

Make sure you have all the files under [ms-swift/swift at main · modelscope/ms-swift] (https://github.com/modelscope/ms-swift/tree/main/swift) at the time of deployment

1.Deploy unfine-tuned models(Swift version is 3.3.0.post1)

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template

model_id_or_path = '/.cache/modelscope/hub/qwen/Qwen1___5-72B-Chat'  # model_id or model_path
system = 'You are a helpful assistant.'

max_new_tokens = 512
temperature = 0
stream = None

engine = PtEngine(model_id_or_path)
template = get_template(engine.model.model_meta.template, engine.tokenizer, default_system=system)

engine.default_template = template
```

2.Deploy fine-tuned model

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template

last_model_checkpoint = 'output/vx-xxx/checkpoint-xxx'
model_id_or_path = '/.cache/modelscope/hub/qwen/Qwen1___5-72B-Chat'  # model_id or model_path
system = 'You are a helpful assistant.'

max_new_tokens = 512
temperature = 0
stream = None

engine = PtEngine(model_id_or_path, adapters=[last_model_checkpoint])
template = get_template(engine.model.model_meta.template, engine.tokenizer, default_system=system)

engine.default_template = template
```

Test performance under different schemes


```python
import os
import time  # 引入时间模块
import pandas as pd


# 数据读取和预处理
data = pd.read_json('data/eval_data.jsonl', lines=True)
selected_data = data[['question', 'label']]
known_labels = list(selected_data['label'].unique())
departments_str = ', '.join(known_labels)
sample_data = selected_data.head(2000)

# 定义提取科室名称的函数
def extract_department(response, known_labels):
    # 按长度从长到短排序已知科室名称，确保更具体的科室名称优先匹配
    known_labels_sorted = sorted(known_labels, key=len, reverse=True)
    for label in known_labels_sorted:
        if label in response:
            return label
    return None

# 初始化统计变量
correct_responses = 0
error1_count = 0  # 分类错误
error2_count = 0  # 未知分类
total_inference_time = 0  # 总推理时间
inference_times = []  # 每轮推理时间

# 打开输出文件
with open('result/result.txt', 'w', encoding='utf-8') as file:
    # 遍历数据
    for index, row in sample_data.iterrows():
        symptom_description = row['question']
        expected_department = row['label']
        
        # 构造查询
        query = f'作为一个高级语言模型，你的任务是根据病人的口述症状来推荐合适的科室。请根据病人的描述“{symptom_description}”，将其分诊到以下科室中的一个：{departments_str}。请注意，只能选择上述提供的科室，不要提及或推荐任何未列出的科室。只需要输出科室，不需要做任何分析。'
        
        # 记录开始时间
        start_time = time.time()
        
        # 发送请求
        request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
        resp_list = engine.infer([infer_request], request_config)
        
        # 记录结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        total_inference_time += inference_time
        
        # 获取响应
        response = resp_list[0].choices[0].message.content if resp_list else ''
        
        # 提取科室名称
        extracted_department = extract_department(response, known_labels)
        
        # 判断是否正确
        if extracted_department == expected_department:
            correct_responses += 1
        else:
            # 判断错误类型
            if extracted_department in known_labels:
                error_type = "分类错误"
                error1_count += 1
            else:
                error_type = "未知分类"
                error2_count += 1
            
            # 写入错误案例
            file.write(f"Question: {symptom_description}\n")
            file.write(f"AI Response: {extracted_department}\n")
            file.write(f"Expected Department: {expected_department}\n")
            file.write(f"Error Type: {error_type}\n")
            file.write(f"Inference Time: {inference_time:.4f} seconds\n")  # 写入当前推理时间
            file.write("-" * 30 + "\n")
    
    # 计算并写入统计结果
    total_samples = len(sample_data)
    accuracy = (correct_responses / total_samples) * 100 if total_samples > 0 else 0
    error1_rate = (error1_count / total_samples) * 100 if total_samples > 0 else 0
    error2_rate = (error2_count / total_samples) * 100 if total_samples > 0 else 0
    avg_inference_time = total_inference_time / total_samples if total_samples > 0 else 0
    
    file.write(f"AI Response Accuracy: {accuracy:.2f}%\n")
    file.write(f"AI 错误分类: {error1_rate:.2f}%\n")
    file.write(f"AI 错分未出现的标签: {error2_rate:.2f}%\n")
    file.write(f"Average Inference Time per Sample: {avg_inference_time:.4f} seconds\n")

print("评估完成，结果已写入文件。")
```

2.Fewshots

```python
with open('result/reslut.txt', 'w', encoding='utf-8') as file:
    # 遍历数据
    for index, row in sample_data.iterrows():
        symptom_description = row['question']
        expected_department = row['label']
        example1 = 'Question: 孩子的肠胃本来就不怎么好，昨天孩子又吃了很多的羊肉串，结果晚上孩子就开始腹泻了，肚子一直不舒服，不停的上厕所，晚上都没有睡好觉，白天孩子还有点腹泻了。请问小孩腹泻的，怎么止泻？Reasoning: 考虑到孩子之前肠胃就存在问题，加上大量食用羊肉串可能导致的消化不良或食物中毒，孩子出现了腹泻和肚子不舒服的症状。这些症状需要儿科医生的评估和治疗，以确定是否需要特定的止泻药物或其他治疗。Expected Department: 儿科'
        example2 = 'Question: 在一次医院体检白带取样时感染了霉菌。二月份开始感染，月经后用了达克林栓剂七天，不适症状消失，但经期的前几天又会不舒服，到现在还是这样的症状。用药的过程中，还伴有出血的症状，血量足够铺满底裤不会打湿，请问医生，我该怎么办?Reasoning: 患者描述了反复的霉菌感染症状，尽管使用了达克林栓剂治疗。出血症状可能表明有更复杂的妇科问题。因此，需要妇产科医生进行全面评估，以排除其他潜在的妇科疾病，并提供适当的治疗方案。Expected Department: 妇产科'
        examples = example1 + '。 ' + example2
        # 构造查询
        query = f'作为一个高级语言模型，你的任务是根据病人的口述症状来推荐合适的科室。请根据病人的描述“{symptom_description}”，将其分诊到以下科室中的一个：{departments_str}。请注意，只能选择上述提供的科室，不要提及或推荐任何未列出的科室。只需要输出科室，不需要做任何分析。以下是两个例子，你只需要输出Expected Department的内容并且不需要输出"Expected Department"：{examples}'
        
        # 记录开始时间
        start_time = time.time()
        
        # 发送请求
        request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        infer_request = InferRequest(messages=[{'role': 'user', 'content': query}])
        resp_list = engine.infer([infer_request], request_config)
```
3.Task pipeline
```python
        # 构造第一个查询（判断疾病类型）
        query1 = f'作为一个高级语言模型，你的任务是根据以下病人的描述来判断病人的疾病类型，只需要输出你判断的疾病名称，不需要任何其他建议，以下是患者的口述：{symptom_description}。请注意只需要输出正确的疾病名称，只需要输出正确的名称，不需要做任何分析。'
        
        # 记录开始时间
        start_time = time.time()
        
        # 发送第一个请求
        request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
        infer_request = InferRequest(messages=[{'role': 'user', 'content': query1}])
        resp_list = engine.infer([infer_request], request_config)
        
        # 记录中间时间
        mid_time = time.time()
        inference_time1 = mid_time - start_time
        
        # 获取响应
        diseases = resp_list[0].choices[0].message.content if resp_list else ''
        
        # 构造第二个查询（推荐科室）
        query2 = f'作为一个高级语言模型，你的任务是根据病人的口述症状和疾病判断来推荐合适的科室。请根据病人的描述“{symptom_description}”疾病判断：“{diseases}”，将其分诊到以下科室中的一个：{departments_str}。请注意，只能选择上述提供的科室，不要提及或推荐任何未列出的科室。只需要输出科室，不需要做任何分析。'
        
        # 发送第二个请求
        infer_request = InferRequest(messages=[{'role': 'user', 'content': query2}])
        resp_list = engine.infer([infer_request], request_config)
        
        # 记录结束时间
        end_time = time.time()
        inference_time2 = end_time - mid_time
        total_inference_time += (inference_time1 + inference_time2)
        inference_times.append(inference_time1 + inference_time2)
        
        # 获取响应
        response = resp_list[0].choices[0].message.content if resp_list else ''
        
        # 提取科室名称
        extracted_department = extract_department(response, known_labels)

```
