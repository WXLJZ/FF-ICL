import numpy as np

def get_csr_struct_features(inputs, tensor_words_list, vehicle_words_list):
    """找出本体或喻体距离比较词的‘像’的距离"""
    vector = []
    for i,sentence in enumerate(inputs):
        comparator_index = sentence.index("像")
        tensor_count = 0
        vehicle_count = 0
        tensor_to_comparator = 0
        vehicle_to_comparator = 0

        for tensor_entry in tensor_words_list[i]:
            tensor_count = len(tensor_words_list[i])
            if sentence.rfind(tensor_entry, 0, comparator_index) != -1:
                tensor_to_comparator = tensor_to_comparator + (comparator_index - sentence.rfind(tensor_entry, 0, comparator_index))
            else:
                tensor_to_comparator = tensor_to_comparator - 20
        tensor_to_comparator = tensor_to_comparator / tensor_count

        for vehicle_entry in vehicle_words_list[i]:
            vehicle_count = len(vehicle_words_list[i])
            if sentence.find(vehicle_entry, comparator_index) != -1:
                vehicle_to_comparator = vehicle_to_comparator + (comparator_index - sentence.find(vehicle_entry, comparator_index))
            else:
                vehicle_to_comparator = vehicle_to_comparator + 20
        vehicle_to_comparator = vehicle_to_comparator / vehicle_count

        vector.append([tensor_to_comparator, tensor_count, vehicle_to_comparator, vehicle_count, len(list(sentence))])

    return vector

def get_cmre_struct_features(inputs, tenor_words_list, vehicle_words_list, span_types):
    '''找出本体距喻体的距离，喻体、喻体属性、喻体部件、喻体动作的数量，语句长度特征'''
    vectors = []
    for i, sentence in enumerate(inputs):
        # 初始化本体距喻体的距离，喻体、喻体属性、喻体部件、喻体动作的数量，语句长度
        tenor_vehicle_distance = 0
        normal_vehicle_count = 0
        attribute_vehicle_count = 0
        part_vehicle_count = 0
        action_vehicle_count = 0
        sent_length = len(list(sentence))

        tenor_words = tenor_words_list[i]
        vehicle_words = vehicle_words_list[i]
        span_type = span_types[i]

        # 遍历各组本体、喻体，做距离计算
        tenor_vehicle_group_count = 0
        for tenor, vehicle in zip(tenor_words, vehicle_words):
            tenor_vehicle_group_count += 1
            tenor_vehicle_distance += sentence.find(tenor) - sentence.find(vehicle)
        tenor_vehicle_distance = tenor_vehicle_distance / tenor_vehicle_group_count

        # 列表去重
        unique_tenor_words = list(set(tenor_words))
        unique_vehicle_words = list(set(vehicle_words))
        # 把span_types转变为字典类型，方便后续查询喻体的类别
        span_types_dict = {word[0]: word[1] for word in span_type}
        # 计算本体数量
        tenor_count = len(unique_tenor_words)
        # 遍历每个句子的喻体，识别每个喻体属于哪个类别的喻体，计算各类别喻体数量
        for element in unique_vehicle_words:
            if span_types_dict[element] == "喻体":
                normal_vehicle_count += 1
            elif span_types_dict[element] == "喻体属性":
                attribute_vehicle_count += 1
            elif span_types_dict[element] == "喻体部件":
                part_vehicle_count += 1
            elif span_types_dict[element] == "喻体动作":
                action_vehicle_count += 1
            else:
                print(f"Error! No key '{element}' in span_types_dict '{span_types_dict}' !")

        vectors.append([tenor_vehicle_distance, tenor_count, normal_vehicle_count, attribute_vehicle_count, part_vehicle_count, action_vehicle_count, sent_length])

    return vectors


def get_lcc_struct_features(inputs, tenor_words_list, vehicle_words_list):
    vector = []
    for i, sentence in enumerate(inputs):
        tenor_vehicle_distance = sentence.find(tenor_words_list[i][0]) - sentence.find(vehicle_words_list[i][0])
        tenor_length = len(tenor_words_list[i][0].split())
        vehicle_length = len(vehicle_words_list[i][0].split())
        sent_length = len(list(sentence))

        vector.append(
            [tenor_vehicle_distance, tenor_length, vehicle_length, sent_length])

    return vector

def euclidean_distance(v1, v2):
    """计算两个向量之间的欧几里得距离"""
    return np.linalg.norm(np.array(v1) - np.array(v2))

def selected_top_k_similarity(matrix_distance, k):
    """对相似度矩阵的每行进行降序排序，选择其中的前k%置1.目的是找出一个batch中相似度最高的一些句子"""
    n = len(matrix_distance)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        row = matrix_distance[i]
        sorted_indices = np.argsort(row)[::-1]
        top_k = int(n * k)
        for j in range(top_k):
            idx = sorted_indices[j]
            similarity_matrix[i][idx] = 1
    return similarity_matrix

def compute_structural_similarity(inputs, outputs, span_types, dataset_name, top_k=0.3):
    """计算结构相似度矩阵，主要考虑本体距喻体的距离，喻体、喻体属性、喻体部件、喻体动作的数量，长度"""
    if len(inputs) != len(outputs):
        AssertionError('The length of inputs and outputs must be the same')

    if dataset_name == 'CMRE':
        tenor_words = [[o[0] for o in os] for os in outputs]
        vehicle_words = [[o[1] for o in os] for os in outputs]
        structural_vector = get_cmre_struct_features(inputs, tenor_words, vehicle_words, span_types)
    elif dataset_name == 'CSR':
        tensor_words = [[o for o in os][0] for os in outputs]
        vehicle_words = [[o for o in os][1] for os in outputs]
        structural_vector = get_csr_struct_features(inputs, tensor_words, vehicle_words)
    elif dataset_name == 'LCC':
        tenor_words = [[o for o in os][0] for os in outputs]
        vehicle_words = [[o for o in os][1] for os in outputs]
        structural_vector = get_lcc_struct_features(inputs, tenor_words, vehicle_words)
    else:
        raise ValueError('The dataset name must be CMRE or CSR')

    n = len(structural_vector)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance = euclidean_distance(structural_vector[i], structural_vector[j])
            if i != j:
                distance = 1 / (1 + distance)
            else:
                distance = 0
            distance_matrix[i][j] = distance

    similarity_matrix = selected_top_k_similarity(distance_matrix,top_k)

    return similarity_matrix


if __name__ == "__main__":
    # 示例

    inputs = [
        "我可以不同意你的观点，但我誓死捍卫你说话的权利。",
        "冻结犯罪嫌疑人的存款、汇款",
        "情趣低俗会腐朽一个人的信念",
        "神到底是创造者还是破坏者？",
        "他的眼睛里，闪烁着理想的光芒。",
        "沟通是打开他心门的一把钥匙",
        "笔端开出第一朵爱情诗的蓓蕾。",
        "我不想被卷进这件事",
        "知识是人类进步的阶梯"
    ]
    outputs = [
        [['捍卫', '你说话的权利']],
        [['冻结', '犯罪嫌疑人的存款、汇款']],
        [['腐朽', '情趣低俗'], ['腐朽', '一个人的信念']],
        [['创造者', '神'], ['破坏者', '神']],
        [['闪烁', '理想的光芒'], ['光芒', '理想']],
        [['钥匙', '沟通'], ['门', '心']],
        [['开出', '笔端'], ['蓓蕾', '爱情诗']],
        [['卷进', '事']],
        [['人类进步的阶梯', '知识'], ['人类进步', '阶梯']]
    ]
    span_types = [
        [['捍卫', '喻体动作'], ['你说话的权利', '本体']],
        [['冻结', '喻体动作'], ['犯罪嫌疑人的存款、汇款', '本体']],
        [['腐朽', '喻体动作'], ['情趣低俗', '本体'], ['一个人的信念', '本体']],
        [['创造者', '喻体'], ['神', '本体'], ['破坏者', '喻体']],
        [['闪烁', '喻体动作'], ['理想的光芒', '本体'], ['光芒', '喻体'], ['理想', '本体']],
        [['钥匙', '喻体'], ['沟通', '本体'], ['门', '喻体部件'], ['心', '本体']],
        [['开出', '喻体动作'], ['笔端', '本体'], ['蓓蕾', '喻体部件'], ['爱情诗', '本体']],
        [['卷进', '喻体动作'], ['事', '本体']],
        [['人类进步的阶梯', '喻体'], ['知识', '本体'], ['人类进步', '本体'], ['阶梯', '喻体']]
    ]

    # inputs = [
    #     "Welcome to the very slippery slope (for some of us it's a cliff!!) of gun ownership !",
    #     "In the U.S., poverty is fattening .",
    #     "It is upon the rock of the Second Amendment where the “privilege” argument in reference to “concealed carry of a firearm”, founders and sinks.",
    #     "The Black population did not heat up the Elections with a call for \"get the rascals out\".",
    #     "Government declined by 900 payroll jobs and the private sector added 4,100.",
    #     "\"Good governance\" includes the state, private sector and civil society as all three sectors are critical for deepening democracy , realizing rights and eradicating poverty."
    # ]
    # outputs = [
    #     [['gun ownership'], ['slippery slope']],
    #     [['poverty'], ['fattening']],
    #     [['Second Amendment'], ['rock']],
    #     [['Elections'], ['heat up']],
    #     [['Government'], ['declined']],
    #     [['democracy'], ['deepening']]
    # ]

    # print(compute_structural_similarity(inputs, outputs, span_types, top_k=0.3))
    print(compute_structural_similarity(inputs, outputs, span_types=None, dataset_name="LCC", top_k=0.3))

