import re
from ..extras.constants import (SKELETON_TOKEN_BASE, SKELETON_FRAME_BREAK, 
                                BODY_PART_ORDER, BODY_PART_TOKENS, JOINT_GROUP_MAP,
                                JOINT_ORDER, JOINT_TOKENS,
                                SKELETON_QUERY_BASE
                                )


def get_skeleton_token_str_woBodyPart_woFramebreak(skeleton_indices):
    frame_strings = []
    for frame_indices in skeleton_indices: # 遍历每一帧
        # 将一帧内的关节索引转换为 <skel_i> 字符串
        joint_str = "".join([SKELETON_TOKEN_BASE.format(i) for i in frame_indices])
        frame_strings.append(joint_str)
    
    # 4. 使用 "换帧符" 连接所有帧
    skeleton_token_str = ''.join(frame_strings)
    return skeleton_token_str

def parse_skeleton_token_str_woBodyPart_woFramebreak(skeleton_token_str):
    """
    从 skeleton token 字符串解析出 skeleton indices
    
    Args:
        skeleton_token_str: 格式化的骨架 token 字符串，如：
            "<skel_1><skel_2>...<skel_16><|frame_break|><skel_0><skel_1>..."
    
    Returns:
        skeleton_indices: List[List[int]], 形状为 (T, J)，T是帧数，J是关节数(17)
    """
    # 预编译正则表达式以提高性能
    skel_pattern = re.compile(r'<skel_(\d+)>')
    matches = skel_pattern.findall(skeleton_token_str)
    
    # 3. 转换为整数列表
    skeleton_indices = [int(match) for match in matches]
        
    return skeleton_indices


def get_skeleton_token_str_woBodyPart(skeleton_indices):
    frame_strings = []
    for frame_indices in skeleton_indices: # 遍历每一帧
        # 将一帧内的关节索引转换为 <skel_i> 字符串
        joint_str = "".join([SKELETON_TOKEN_BASE.format(i) for i in frame_indices])
        frame_strings.append(joint_str)
    
    # 4. 使用 "换帧符" 连接所有帧
    skeleton_token_str = SKELETON_FRAME_BREAK.join(frame_strings)
    return skeleton_token_str

def parse_skeleton_token_str_woBodyPart(skeleton_token_str):
    """
    从 skeleton token 字符串解析出 skeleton indices
    
    Args:
        skeleton_token_str: 格式化的骨架 token 字符串，如：
            "<skel_1><skel_2>...<skel_16><|frame_break|><skel_0><skel_1>..."
    
    Returns:
        skeleton_indices: List[List[int]], 形状为 (T, J)，T是帧数，J是关节数(17)
    """
    # 1. 按帧分割字符串
    frame_strings = skeleton_token_str.split(SKELETON_FRAME_BREAK)
    
    skeleton_indices = []
    
    # 预编译正则表达式以提高性能
    skel_pattern = re.compile(r'<skel_(\d+)>')
    
    for frame_str in frame_strings:
        if not frame_str.strip():  # 跳过空字符串
            continue
        
        # 2. 使用正则表达式匹配所有 <skel_数字> 模式
        matches = skel_pattern.findall(frame_str)
        
        # 3. 转换为整数列表
        frame_indices = [int(match) for match in matches]
        
        # 4. 验证关节数量是否正确（应该是17个关节）
        if len(frame_indices) != 17:
            print(f"WARNING: Expected 17 joints per frame, but got {len(frame_indices)} in frame: {frame_str}")
        
        skeleton_indices.append(frame_indices)
    
    return skeleton_indices


def get_skeleton_token_str_wBodyPart(skeleton_indices):
    frame_strings = []
    for frame_indices in skeleton_indices: # 遍历每一帧
        part_strings = []
        # 3. 按照预定义的身体部位顺序进行遍历
        for part_name in BODY_PART_ORDER:
            start_token, end_token = BODY_PART_TOKENS[part_name]
            joint_indices_for_part = JOINT_GROUP_MAP[part_name]
            
            # 提取该部位对应的关节词元
            joint_tokens = [SKELETON_TOKEN_BASE.format(frame_indices[j]) for j in joint_indices_for_part]
            
            # 构建部位字符串: <torso><skel_1><skel_2></torso>
            part_strings.append(start_token + "".join(joint_tokens) + end_token)
        
        # 将一帧内所有部位的字符串连接起来
        frame_strings.append("".join(part_strings))
    
    # 4. 使用 "换帧符" 连接所有帧
    skeleton_token_str = SKELETON_FRAME_BREAK.join(frame_strings)
    return skeleton_token_str


def parse_skeleton_token_str_wBodyPart(skeleton_token_str):
    """
    从 skeleton token 字符串解析出 skeleton indices
    
    Args:
        skeleton_token_str: 格式化的骨架 token 字符串，如：
            "<torso><skel_1><skel_2></torso><left_arm><skel_3></left_arm>...<|frame_break|><torso>..."
    
    Returns:
        skeleton_indices: List[List[int]], 形状为 (T, J)，T是帧数，J是关节数(17)
    """
    # 1. 按帧分割字符串
    frame_strings = skeleton_token_str.split(SKELETON_FRAME_BREAK)
    
    skeleton_indices = []
    
    for frame_str in frame_strings:
        if not frame_str.strip():  # 跳过空字符串
            continue
            
        # 初始化当前帧的关节索引数组，17个关节
        frame_indices = [0] * 17
        
        # 2. 按照身体部位顺序解析每一帧
        remaining_str = frame_str
        
        for part_name in BODY_PART_ORDER:
            start_token, end_token = BODY_PART_TOKENS[part_name]
            joint_indices_for_part = JOINT_GROUP_MAP[part_name]
            
            # 3. 查找当前部位的开始和结束标记
            start_pos = remaining_str.find(start_token)
            end_pos = remaining_str.find(end_token)
            
            if start_pos == -1 or end_pos == -1:
                raise ValueError(f"Could not find {part_name} tokens in frame: {frame_str}")
            
            # 4. 提取部位内容
            part_content = remaining_str[start_pos + len(start_token):end_pos]
            
            # 5. 解析关节词元
            joint_tokens = []
            pos = 0
            while pos < len(part_content):
                # 查找 <skel_数字> 模式
                if part_content[pos:pos+6] == "<skel_":
                    # 找到结束的 >
                    end_bracket = part_content.find(">", pos + 6)
                    if end_bracket == -1:
                        raise ValueError(f"Invalid skeleton token format in {part_content[pos:]}")
                    
                    # 提取数字
                    token_str = part_content[pos:end_bracket + 1]  # 包含完整的 <skel_数字>
                    try:
                        # 从 <skel_123> 中提取 123
                        number = int(token_str[6:-1])
                        joint_tokens.append(number)
                    except ValueError:
                        raise ValueError(f"Invalid skeleton token: {token_str}")
                    
                    pos = end_bracket + 1
                else:
                    pos += 1
            
            # 6. 将解析出的关节索引填入对应位置
            if len(joint_tokens) != len(joint_indices_for_part):
                raise ValueError(f"Mismatch in joint count for {part_name}: expected {len(joint_indices_for_part)}, got {len(joint_tokens)}")
            
            for i, joint_idx in enumerate(joint_indices_for_part):
                frame_indices[joint_idx] = joint_tokens[i]
            
            # 7. 从剩余字符串中移除已处理的部分
            remaining_str = remaining_str[end_pos + len(end_token):]
        
        skeleton_indices.append(frame_indices)
    
    return skeleton_indices


def get_skeleton_token_str_wJoint(skeleton_indices):
    """
    将 skeleton_indices 转换为带有独立关节标签的字符串。
    
    Args:
        skeleton_indices: List[List[int]], 形状为 (T, 17)
    
    Returns:
        skeleton_token_str: 格式化的字符串，如：
            "<Hips><skel_1></Hips><Right_Hip><skel_2></Right_Hip>...<|frame_break|>..."
    """
    frame_strings = []
    for frame_indices in skeleton_indices:  # 遍历每一帧
        joint_strings = []
        # 遍历17个关节
        for i, joint_name in enumerate(JOINT_ORDER):
            start_token, end_token = JOINT_TOKENS[joint_name]
            
            # 获取当前关节的 VQ-VAE 索引
            vq_index = frame_indices[i]
            
            # 构建关节字符串: <Hips><skel_123></Hips>
            joint_str = start_token + SKELETON_TOKEN_BASE.format(vq_index) + end_token
            joint_strings.append(joint_str)
        
        # 将一帧内所有关节的字符串连接起来
        frame_strings.append("".join(joint_strings))
    
    # 使用 "换帧符" 连接所有帧
    skeleton_token_str = SKELETON_FRAME_BREAK.join(frame_strings)
    return skeleton_token_str


def parse_skeleton_token_str_wJoint(skeleton_token_str):
    """
    从带有独立关节标签的字符串解析出 skeleton_indices。
    
    Args:
        skeleton_token_str: 格式化的字符串，如：
            "<Hips><skel_1></Hips><Right_Hip><skel_2></Right_Hip>...<|frame_break|>..."
    
    Returns:
        skeleton_indices: List[List[int]], 形状为 (T, 17)
    """
    # 1. 按帧分割字符串
    frame_strings = skeleton_token_str.split(SKELETON_FRAME_BREAK)
    
    skeleton_indices = []
    skel_pattern = re.compile(r'<skel_(\d+)>')
    
    for frame_str in frame_strings:
        if not frame_str.strip():  # 跳过空字符串
            continue
            
        frame_indices = [0] * 17
        
        # 2. 按照关节顺序解析每一帧
        for i, joint_name in enumerate(JOINT_ORDER):
            start_token, end_token = JOINT_TOKENS[joint_name]
            
            # 3. 查找当前关节的开始和结束标记
            start_pos = frame_str.find(start_token)
            end_pos = frame_str.find(end_token)
            
            if start_pos == -1 or end_pos == -1:
                print(f"WARNING!!! Could not find tags for joint '{joint_name}' in frame: {frame_str}")
            
            # 4. 提取关节内容
            content = frame_str[start_pos + len(start_token):end_pos]
            
            # 5. 解析 <skel_...> 词元
            match = skel_pattern.search(content)
            if not match:
                print(f"WARNING!!! Could not find skeleton token inside '{content}' for joint '{joint_name}'")
            
            # 6. 将解析出的关节索引填入对应位置
            frame_indices[i] = int(match.group(1))
        
        skeleton_indices.append(frame_indices)
    
    return skeleton_indices
