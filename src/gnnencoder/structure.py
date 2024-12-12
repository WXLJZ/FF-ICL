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


