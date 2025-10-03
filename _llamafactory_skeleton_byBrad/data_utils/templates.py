
PROMPT_TEMPLATES = {
    'Vid2Skel': {
        'fixed': [
            "Generate the skeleton sequence for the video <video>."
        ],
        'simple': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Generate the skeleton sequence for the video <video>.",
            "Transcribe the actions in <video> into a sequence of skeleton tokens.",
            "What is the skeleton token sequence for the person in <video>?",
            "You are a motion analysis expert. Analyze the video <video> and output the corresponding motion data.",
            "Here is a video of a person moving: <video>. The corresponding skeletal representation is:",
            "Analyze the person's movement in <video> and represent it using skeleton tokens.",

            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Please provide the skeletal representation for the movement in the video <video>.",
        ],
        'bodypart_aware': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Generate the skeleton sequence for the video <video>. Please structure the output for each frame using body part tags (e.g., <torso>...</torso>, <left_arm>...</left_arm>, etc.).",
            "Transcribe the actions in <video> into a sequence of skeleton tokens, ensuring the output is formatted by body parts for each frame.",
            "What is the skeleton token sequence for the person in <video>? Please provide the answer in a structured format with body part tags.",
            "You are a motion analysis expert. Analyze the video <video> and output the corresponding motion data. Your output must be structured frame-by-frame, with each frame containing body part sections like <torso>...</torso>.",
            "Here is a video of a person moving: <video>. The corresponding structural representation, organized by body parts, is:",
            "Analyze the person's movement in <video> and represent it structurally using body part tags.",

            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Please provide the skeletal representation for the movement in the video <video>. Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
            "What would the motion capture data for the video <video> look like? Output the data using the precise format for each frame: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'BodypartAwareExplicit': [
            "Please provide the skeletal representation for the movement in the video <video>. Use <|frame_break|> to separate frames. Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'BodypartAwareExplicit_text': [
            "Please describe the motion of the person in the video <video> using skeleton tokens. Your response should be formatted as: \"Frame 1: torso: ... left_arm: ... right_arm: ... left_leg: ... right_leg: ...\nFrame 2: ... \""
        ],
        'JointAwareExplicit': [
            "Please provide the skeletal representation for the movement in the video <video>. Use <|frame_break|> to separate frames. "
            "Your response must be structured for each frame using all 17 joint tags: <Hips>...</Hips><Right_Hip>...</Right_Hip><Right_Knee>...</Right_Knee><Right_Foot>...</Right_Foot><Left_Hip>...</Left_Hip><Left_Knee>...</Left_Knee><Left_Foot>...</Left_Foot><Spine>...</Spine><Thorax>...</Thorax><Neck/Nose>...</Neck/Nose><Head>...</Head><Left_Shoulder>...</Left_Shoulder><Left_Elbow>...</Left_Elbow><Left_Wrist>...</Left_Wrist><Right_Shoulder>...</Right_Shoulder><Right_Elbow>...</Right_Elbow><Right_Wrist>...</Right_Wrist>.",
        ],
    },
    'SkelPred': {
        'fixed': [
            "Predict the future motion based on the provided skeleton sequence <skeleton>.",
        ],
        'simple': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Continue the motion sequence provided in <skeleton>.",
            "Predict the future motion based on the provided skeleton sequence <skeleton>.",
            "Given the motion <skeleton>, what happens next?",
            "You are a motion prediction expert. Analyze the past motion <skeleton> and output the most likely future motion.",
            "Here is the beginning of a motion sequence: <skeleton>. The continuation of the motion is:",
            "Generate the next set of skeleton tokens that logically follow this sequence: <skeleton>.",

            # --- 完整格式说明 (作为清晰的"锚点") ---
            "Past motion: <skeleton>. Future motion:",
        ],
        'bodypart_aware': [
            # --- 简洁格式说明 ---
            "Continue the motion sequence provided in <skeleton>. The predicted motion should follow the same body part structure (e.g., <torso>...</torso>).",
            "Predict the future motion based on the provided skeleton sequence <skeleton>. Generate the next set of skeleton tokens using the established body part format.",
            "Given the motion <skeleton>, what happens next? Please provide the answer in the same structural format.",
            "You are a motion prediction expert. Analyze the past motion <skeleton> and output the most likely future motion, maintaining the structural format.",
            "Here is the beginning of a motion sequence: <skeleton>. The continuation of the motion, maintaining the same structure, is:",
            
            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Generate the next set of skeleton tokens that logically follow this sequence: <skeleton>. Ensure the output adheres to the full structure: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
            "Past motion: <skeleton>. Future motion (formatted with all body part tags):",
        ],
        'BodypartAwareExplicit': [
            "Generate the next set of skeleton tokens that logically follow this sequence: <skeleton>. Use <|frame_break|> to separate frames. Ensure the output adheres to the full structure: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'BodypartAwareExplicit_text': [
            "Here's a motion sequence represented using skeleton tokens: \n<skeleton> \nPredict the future motion using skeleton tokens that has the same number of frames as the history motion. Your response should be formatted as: \"Future Frame 1: torso: ... left_arm: ... right_arm: ... left_leg: ... right_leg: ...\nFuture Frame 2: ... \""
        ],
        'JointAwareExplicit': [
            # TODO
        ],
    },
    'Text2Skel': {
        'fixed': [
            "Generate the skeleton motion for the description: \"<text_description>\".",
        ],
        'simple': [
            # --- 简洁格式说明 (大部分采用此形式) ---
            "Generate the skeleton sequence for the following description: <text_description>.",
            "Create a motion sequence based on this text: \"<text_description>\".",
            "What would the motion for '<text_description>' look like in skeleton tokens?",
            "You are a choreographer. Animate the following description into a skeleton sequence: <text_description>.",
            "Description: <text_description>. Corresponding skeleton sequence:",
            "Convert the following text into skeleton motion: \"<text_description>\".",

            # --- 完整格式说明 (作为清晰的"锚点") ---
            "Generate the skeleton motion for the description: \"<text_description>\".",
        ],
        'bodypart_aware': [
            # --- 简洁格式说明 ---
            "Generate the skeleton sequence for the following description: <text_description>. Please structure the output using body part tags (e.g., <torso>...</torso>).",
            "Create a motion sequence based on this text: \"<text_description>\". Ensure the output is formatted with body part tags for each frame.",
            "What would the motion for '<text_description>' look like in skeleton tokens? Please provide the answer in the structured body part format.",
            "You are a choreographer. Animate the following description into a skeleton sequence: <text_description>. Use the standard body part structure for your output.",
            "Description: <text_description>. Corresponding skeleton sequence:",

            # --- 完整格式说明 (作为清晰的“锚点”) ---
            "Generate the skeleton motion for the description: \"<text_description>\". Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'BodypartAwareExplicit': [
            "Generate the skeleton motion for the description: \"<text_description>\". Use <|frame_break|> to separate frames. Your response must be structured for each frame using all five body part tags: <torso>...</torso><left_arm>...</left_arm><right_arm>...</right_arm><left_leg>...</left_leg><right_leg>...</right_leg>.",
        ],
        'JointAwareExplicit': [
            # TODO
        ],
    },
}

TASK_TEMPLATE = {
    'Vid2Skel': {
        "conversations": [{"from": "human", "value": None},
                          {"from": "gpt", "value": "<skeleton>"}],
        "videos": [],
        "skeletons": None
    },
    'SkelPred': {
        "conversations": [{"from": "human", "value": None},
                          {"from": "gpt", "value": "<skeleton>"}],
        "videos": [],
        "skeletons": None
    },
    'Text2Skel': {
        "conversations": [{"from": "human", "value": None},
                          {"from": "gpt", "value": "<skeleton>"}],
        "videos": [],
        "skeletons": None # 只包含一个目标文件
    }
}

