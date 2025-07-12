import heapq
from collections import defaultdict, Counter
import pathlib

def read_corpus(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [list(word) for line in f for word in line.strip().split()]

def run_train_bpe(input_path, vocab_size, special_tokens):
    # 初始化语料和词汇
    corpus = read_corpus(input_path)
    vocab = {token: token for token in special_tokens}
    merges = []
    
    # 初始字符对计数
    pairs_counter = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pairs_counter[word[i], word[i+1]] += 1
    
    # 创建最大堆（使用负频率）
    heap = [(-freq, pair) for pair, freq in pairs_counter.items()]
    heapq.heapify(heap)
    
    # 辅助函数：减少字符对计数
    def decrement_pair(a, b):
        pair = (a, b)
        if pairs_counter[pair] > 1:
            pairs_counter[pair] -= 1
        else:
            del pairs_counter[pair]
    
    # 主训练循环
    while len(vocab) < vocab_size and heap:
        # 惰性删除：跳过无效条目
        while heap:
            neg_freq, best_pair = heapq.heappop(heap)
            freq = -neg_freq
            if best_pair in pairs_counter and pairs_counter[best_pair] == freq:
                break
        else:  # 堆为空
            break
        
        # 创建新符号
        new_token = ''.join(best_pair)
        vocab[new_token] = new_token
        merges.append(best_pair)
        
        # 更新语料和字符对计数
        a, b = best_pair
        for word in corpus:
            i = 0
            while i < len(word) - 1:
                # 检查当前字符对
                if word[i] == a and word[i+1] == b:
                    # 获取相邻字符
                    left = word[i-1] if i > 0 else None
                    right = word[i+2] if i+2 < len(word) else None
                    
                    # 减少旧字符对计数
                    decrement_pair(a, b)
                    if left:
                        decrement_pair(left, a)
                    if right:
                        decrement_pair(b, right)
                    
                    # 执行合并
                    word[i:i+2] = [new_token]
                    
                    # 增加新字符对计数
                    if left:
                        new_pair_left = (left, new_token)
                        pairs_counter[new_pair_left] += 1
                        heapq.heappush(heap, (-pairs_counter[new_pair_left], new_pair_left))
                    if right:
                        new_pair_right = (new_token, right)
                        pairs_counter[new_pair_right] += 1
                        heapq.heappush(heap, (-pairs_counter[new_pair_right], new_pair_right))
                    
                    # 跳过新合并的符号
                    i += 1 if left is None else 0
                else:
                    i += 1
    
    return vocab, merges