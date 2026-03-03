import numpy as np
import base64
import io
import json
import pandas as pd
import pickle
import csv
import re
import os.path as osp


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    return ret


def prepare_itlist(inputs, image_size=-1,  **kwargs):
    assert np.all([isinstance(x, dict) for x in inputs])
    has_images = np.sum([x['type'] == 'image' for x in inputs])
    if has_images:
        content_list = []
        for msg in inputs:
            if msg['type'] == 'text':
                content_list.append(dict(type='text', text=msg['value']))
            elif msg['type'] == 'image':
                from PIL import Image
                img = Image.open(msg['value'])
                b64 = encode_image_to_base64(img, target_size=image_size)
                img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail='high')
                content_list.append(dict(type='image_url', image_url=img_struct))
    else:
        assert all([x['type'] == 'text' for x in inputs])
        text = '\n'.join([x['value'] for x in inputs])
        content_list = [dict(type='text', text=text)]
    return content_list


def prepare_inputs(inputs, system_prompt=None, **kwargs):
    input_msgs = []
    if system_prompt is not None:
        input_msgs.append(dict(role='system', content=system_prompt))
    assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
    if 'role' in inputs[0]:
        assert inputs[-1]['role'] == 'user', inputs[-1]
        for item in inputs:
            input_msgs.append(dict(role=item['role'], content=prepare_itlist(item['content'], **kwargs)))
    else:
        input_msgs.append(dict(role='user', content=prepare_itlist(inputs, **kwargs)))
    return input_msgs


def find_image(output_dir, index):
    for suffix in ['png', 'jpg', 'jpeg']:
        img_path = osp.join(output_dir, f"{index}.{suffix}")
        if osp.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Cannot find output images {index} in {output_dir}!!!")


def indiactor_average(df):
    score_cols = [
        "scores_appearance_consistency_all",
        "scores_perceptual_quality_all",
        "scores_semantic_consistency_all",
        "scores_logical_coherence_all",
        "scores_scientific_plausibility_all",
        "scores_process_plausibility_all"
    ]
    fdf = df.copy()
    fdf[score_cols] = fdf[score_cols].apply(pd.to_numeric, errors="coerce")
    fdf[score_cols] = fdf[score_cols].replace(0, np.nan)
    indicator_average_scores = fdf[score_cols].mean(skipna=True)
    indicator_average_scores_100 = (indicator_average_scores - 1) * 25
    overall_average = indicator_average_scores_100.mean()
    return indicator_average_scores_100, overall_average


def accuracy_overall(df):
    cols_5 = [
        "scores_appearance_consistency_all",
        "scores_perceptual_quality_all",
        "scores_semantic_consistency_all",
        "scores_logical_coherence_all",
        "scores_scientific_plausibility_all"
    ]
    cols_4 = [
        "scores_appearance_consistency_all",
        "scores_perceptual_quality_all",
        "scores_semantic_consistency_all",
        "scores_logical_coherence_all"
    ]
    if "index" not in df.columns:
        raise ValueError("The DataFrame must contain an 'index' column.")
    fdf = df.copy()
    cat = fdf["index"].astype(str).str.split("_").str[0]
    fdf = fdf[~cat.isin(["process"])]
    fdf[cols_5] = fdf[cols_5].apply(pd.to_numeric, errors="coerce")
    mask_ds = cat.isin(["dynamic", "scientific"])
    mask_st = cat.isin(["state", "temporal"])
    perfect_mask = pd.Series(False, index=fdf.index)
    perfect_mask.loc[mask_ds] = (fdf.loc[mask_ds, cols_5] == 5).all(axis=1)
    perfect_mask.loc[mask_st] = (fdf.loc[mask_st, cols_4] == 5).all(axis=1)
    total = len(fdf)
    perfect_count = int(perfect_mask.sum())
    ratio = perfect_count / total if total > 0 else 0.0
    perfect_indices = fdf.loc[perfect_mask, "index"].tolist()
    return ratio*100, perfect_indices

def calculate_bootstrap_ci(file_path, n_bootstraps=10000, confidence_level=0.95):
    df = pd.read_excel(file_path, engine='openpyxl')
    metric_cols = ['scores_appearance_consistency_all', 'scores_perceptual_quality_all', 'scores_semantic_consistency_all', 'scores_logical_coherence_all', 'scores_scientific_plausibility_all', 'scores_process_plausibility_all']

    for col in metric_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' is not in the Excel file.")
            return
    df[metric_cols] = df[metric_cols].replace(0, np.nan)

    # Data Slices
    # 0:dynamic, 1:simulation, 2:state, 3:temporal, 4:sub_state (process)
    slice_names = ['dynamic', 'simulation', 'state', 'temporal', 'sub_state']
    slices = {
        'dynamic': slice(0, 65),
        'simulation': slice(65, 88),
        'state': slice(88, 137),
        'temporal': slice(137, 203),
        'sub_state': slice(203, 237)
    }
    total_rows = len(df)
    if total_rows != 237:
        print(f"Warning: The file has {total_rows} lines, expected 237 lines. Please confirm whether the slicing range is applicable.")
    indices_map = {name: np.arange(slices[name].start, slices[name].stop) for name in slice_names}
    col_app = df['scores_appearance_consistency_all'].values
    col_perc = df['scores_perceptual_quality_all'].values
    col_sem = df['scores_semantic_consistency_all'].values
    col_logic = df['scores_logical_coherence_all'].values
    col_sci = df['scores_scientific_plausibility_all'].values
    col_proc = df['scores_process_plausibility_all'].values

    bootstrap_results_raw = []
    # Bootstrap
    for i in range(n_bootstraps):
        # [0]: dynamic, [1]: simulation, [2]: state, [3]: temporal, [4]: sub_state
        resampled_indices_list = []
        for name in slice_names:
            idxs = indices_map[name]
            sample = np.random.choice(idxs, size=len(idxs), replace=True)
            resampled_indices_list.append(sample)
        # dynamic + simulation + state + temporal --> app, perc, sem, logic
        general_indices = np.concatenate(resampled_indices_list[0:4])
        mu_app = np.nanmean(col_app[general_indices])
        mu_perc = np.nanmean(col_perc[general_indices])
        mu_sem = np.nanmean(col_sem[general_indices])
        mu_logic = np.nanmean(col_logic[general_indices])
        # dynamic + simulation --> sci
        sci_indices = np.concatenate([resampled_indices_list[0], resampled_indices_list[1]])
        mu_sci = np.nanmean(col_sci[sci_indices])
        # sub_state --> proc
        proc_indices = resampled_indices_list[4]
        mu_proc = np.nanmean(col_proc[proc_indices])
        # --- Overall ( 1-5 ) ---
        overall_score_raw = (mu_app + mu_perc + mu_sem + mu_logic + mu_sci + mu_proc) / 6
        bootstrap_results_raw.append(overall_score_raw)
    bootstrap_results_raw = np.array(bootstrap_results_raw)

    # (Mean_1_to_5 - 1) * 25
    bootstrap_results_100 = (bootstrap_results_raw - 1) * 25
    sorted_scores = np.sort(bootstrap_results_100)
    alpha = (1 - confidence_level) / 2
    lower_idx = int(n_bootstraps * alpha)
    upper_idx = int(n_bootstraps * (1 - alpha))
    ci_lower = sorted_scores[lower_idx]
    ci_upper = sorted_scores[upper_idx]

    return ci_lower, ci_upper

def extract(answer):
    matches = re.findall(r'\*{0,2}Final Score(?:s)?[:：]?\s*\*{0,2}[:：]?\s*([\d\s,]+)\*{0,2}', answer, re.IGNORECASE)
    numbers = []
    if matches:
        for match in matches:
            extracted_numbers = re.findall(r'\d+', match.replace('\n', ' '))
            if extracted_numbers:
                numbers.extend(map(int, extracted_numbers))
                break
        if numbers != []:
            return numbers

    else:
        return None


prompt_appearance_consistency = """You are a professional image appearance evaluation expert, skilled at judging appearance consistency across multiple images. You will receive the following input:
-	Image A: Consists of two parts. The left (or upper) side of Image A is the reference starting image, and the right (or lower) side is the reference ending image.
-	Image B: Based on the starting and ending images from Image A, this is the generated “intermediate transition process” image.
-	Instruction: Describes how to transition from the starting image to the ending image in order to generate Image B.

Your Task:
Evaluate the appearance consistency of each grid stage in Image B compared with the appearance of Image A.

Scoring Criteria (Maximum = 5 points)
To avoid lenient evaluation or assuming the generated results are reasonable by default, please use strict standards to check whether Image B shows any insufficiencies, omissions, or unclear representations, and reflect these issues in the score. Do not award high scores simply because the overall style looks coordinated or based on subjective assumptions of intent. Scoring must follow the most rigorous and conservative judgment.
-	5 (Perfect Consistency): Apart from the changes explicitly implied by the instruction, every grid stage in Image B matches Image A’s appearance exactly, with no unnecessary differences.
-	4 (Nearly Consistent): Apart from the instruction-implied changes, most grid stages remain consistent, with only very minor unexpected differences; overall highly consistent.
-	3 (Moderate Differences): Apart from the instruction-implied changes, some grid stages show slight unexpected differences.
-	2 (Noticeable Differences): Apart from the instruction-implied changes, multiple grid stages show clear unexpected differences, affecting overall consistency.
-	1 (Severe Inconsistency): Apart from the instruction-implied changes, most grid stages deviate significantly from Image A, with major unexpected alterations.

Notes:
-	Ignore the grid structure itself (e.g., grid lines, separation effect, numbering). Do not consider these as style differences. Only focus on the visual appearance of each stage within the grid.
-	Ignore content changes explicitly implied by the instruction. Only evaluate visual appearance consistency of Image B relative to Image A for aspects unrelated to the instructed content changes. Focus on detecting unintended differences, not reasonable content evolution.
-	Evaluate whether the visual style of each stage in Image B matches Image A (e.g., realistic, floral, cartoon, etc.).

Input:
-	Image A: The first uploaded photo.
-	Image B: The second uploaded photo.
-	Instruction: {Instruction}

Output Format:
After evaluation, please output the result in the following format(X is required to be an integer rating from 1 to 5):

Final Score: X
"""


prompt_perceptual_quality = """You are a professional image quality evaluation expert, specializing in analyzing the perceptual quality of images based on visual perception standards. You will receive the following input:
-	Image A: Image A describes the intermediate transition stages between a reference starting image and a reference ending image.

Your Task:
Evaluate the perceptual quality of each grid stage in Image A.

Notes:
-	Ignore the influence of grid division itself. Do not treat grid structures (e.g., grid lines, separation effects, numbering) as quality issues. Also ignore any quality issues that arise solely from grid formatting. Focus only on the perceptual quality of each grid stage within Image A.
-	Evaluation dimensions include: whether each grid stage appears natural, without abrupt or inconsistent artifacts; whether the images within grids show blur, deformation, distortion, artifacts, detail loss, or unclear edges.

Scoring Criteria (Maximum = 5 points)
To avoid lenient evaluation or assuming generated results are inherently reasonable, please use strict standards to examine whether Image A shows any insufficiencies, omissions, or unclear representations, and reflect them in the score. Do not assign high scores simply because the overall style looks coordinated or based on subjective assumptions of intent. Scoring must follow the most rigorous and conservative judgment.
-	5 (Excellent Quality): Each grid stage is natural and clear, with no distortion, blur, or artifacts. Overall visual effect is excellent.
-	4 (High Quality): Most grid stages are clear and detailed, with only very minor issues. Overall quality remains high.
-	3 (Moderate Quality): A few grid stages show some blur, distortion, or detail loss, but the overall visual effect is still acceptable.
-	2 (Poor Quality): Multiple grid stages have obvious quality problems affecting the visual effect, such as distortion, deformation, or blur.
-	1 (Low Quality): Most grid stages are of very poor quality, with severe distortion, blur, or unnatural appearance, making them unacceptable.

Input:
-	Image A: The first uploaded photo.

Output Format:
After completing the evaluation, please output the result in the following format(X is required to be an integer rating from 1 to 5):

Final Score: X
"""

prompt_semantic_consistency = """You are a professional image evaluation expert, responsible for strictly judging whether a "multi-stage process image" accurately complies with the given generation instruction. Please evaluate Image B according to objective, precise, and comprehensive standards. You will receive the following information:
- Image A: This image consists of two parts. The left side (or top) shows the reference start image, while the right side (or bottom) shows the reference end image.
- Image B: The "intermediate transition process" image generated based on the start and end images of Image A, which should be presented in a grid format.
- Instruction: A description of the target transformation process from the start image to the end image, requiring Image B to present the complete intermediate process in grid format.

Evaluation principles:
- Independence: Assessment must rely solely on the explicit content of Image B, without using Image A to infer or fill in missing information.
- Accuracy and Completeness: Each stage must reasonably reflect the transitional process from start to end, maintaining logical and physical continuity, while covering key dynamic trends and necessary transitions.
- Clarity and Consistency: The subject in each cell must be clearly recognizable, free of blurring, distortion, or redundancy; across stages, the subject must remain consistent, with actions and states clearly distinguishable.
- Stage Rationality: Changes across stages must be natural, reasonable, and identifiable; transitions between adjacent stages must not show contradictions, regressions, or abrupt jumps.
- Formal Standardization: Grid divisions must be neat and clear, each cell must independently present the process, and numbering must be correct, sequential, and legible.

Task requirements:
- Based on Image A and the instruction, infer the complete intermediate transition steps and describe them clearly.
- Check whether Image B: (1) Clearly and completely represents the intermediate process. (2) Maintains subject consistency. (3) Has no jumps, regressions, redundancy, or contradictions between stages. (4) Covers the main dynamic trends and key transitional stages. (5) Has standardized grid division with clear layout. (6) Uses continuous, clear numbering without omissions or errors.
- Every identified issue must result in a score deduction.

Scoring criteria (maximum score is 5):
To avoid overly lenient evaluations or default assumptions that the generated result is reasonable, you must apply strict standards to review whether Image B contains any deficiencies, omissions, or unclear expressions, and reflect these clearly in the score. Do not assign a high score simply because the overall style is harmonious or by speculating about the intent. Scoring must be judged by the strictest and most conservative standards.
- 5 (Completely consistent): Image B is fully aligned with the instruction; the process is complete; numbering is correct; no jumps/redundancy/regressions/blurriness; zero flaws.
- 4 (Almost consistent): Overall highly aligned, with only minor issues (e.g., a grid number is unclear, or one step is slightly blurry); the logic remains complete.
- 3 (Moderate differences): Multiple issues are present (e.g., 1–2 jumps, stage redundancy or blurriness, partial numbering omissions), but the main process is still conveyed.
- 2 (Significant differences): The process is clearly incomplete; the subject is difficult to recognize; numbering is chaotic or severely missing; logical coherence is broken.
- 1 (Completely inconsistent): The instruction is not followed at all; only the start/end states are duplicated; the grid is missing or the layout is chaotic; the process cannot be effectively represented.

Example explanation:

- “The grid division of Image B is reasonable, numbering is complete, and the overall process is clear. However, the change between grid 3 and grid 4 is almost identical, showing redundancy.”
→ Final Score: 4

- “Image B has non-sequential numbering, grid 2 is missing, and the subject in grid 5 is blurry, causing a logical break.”
→ Final Score: 2

Input:
- Image A: The first uploaded photo.
- Image B: The second uploaded photo.
- Instruction: {Instruction}

Output format:
After completing the evaluation, please output the result as follows(X is required to be an integer rating from 1 to 5):

Final Score: X
"""

prompt_logical_coherence = """You are a transition logic evaluation expert, specializing in analyzing whether the processes shown in images demonstrate reasonable transition logic. You will receive the following input:
-	Image A: Image A consists of two parts. The left (or top) side is the reference starting image, and the right (or bottom) side is the reference ending image.
-	Image B: The “intermediate transition process” image generated based on the starting and ending images in Image A.
-	Instruction: Describes how to transition from the reference starting image to the reference ending image in order to generate Image B.

Your Task:
Evaluate the reasonableness and naturalness of the transition logic between stages in Image B.

Scoring Criteria (Maximum = 5 points)
To prevent lenient evaluations or assuming generated results are inherently reasonable, please apply strict standards when examining Image B for deficiencies, omissions, or unclear aspects, and reflect these in the score. Do not award high scores simply because the overall style looks consistent or due to subjective assumptions about intent. Scores must be judged by the most rigorous and conservative standards.
-	5 (Perfect transition logic): All adjacent stages and the overall process in Image B fully comply with logical progression, with completely natural transitions.
-	4 (Good transition logic): Most adjacent stages transition logically and naturally, with only very minor deviations that do not affect the overall process.
-	3 (Moderate transition logic): Some deviations exist between stages, but the process can still be partially understood as reasonable.
-	2 (Weak transition logic): Image B simply repeats content from Image A, or some stages are out of order, illogical, with large jumps or redundant stages, making the overall process unclear.
-	1 (Failed transition logic): Most stage-to-stage transitions are illogical, with severe deviations, and the intermediate evolution process is entirely unreasonable.

Guidelines:
-	Stage grid order confirmation: If Image B includes stage numbering that is continuous, sequential, and easy to recognize, evaluate adjacent stages strictly based on numbering. Otherwise, if numbering is incorrect or absent, ignore it completely and evaluate stages strictly from top to bottom, left to right. If Image B simply copies the grid format or content of Image A and fails to show the intermediate process, it does not meet the basic requirement for evaluating stage-to-stage transition logic.
-	Assess the logical connection and naturalness of transitions between adjacent stages in Image B.
-	Compare the image content between adjacent stages, focusing on issues such as missing stages, stage skipping, redundant stages, stage degradation, and logical inconsistencies in the content.
-	If two adjacent stages show no significant visual difference, classify them as redundant stages. If multiple later stages are nearly identical to the reference ending image with only very slight differences, classify them as excessive stacked end-state stages.

Example:
“Image B is reasonably divided into grids, but the numbering labels are inaccurate. Following the order from top to bottom and left to right, the transitions between adjacent stages show minor logical issues. A few adjacent stages are nearly repetitive, leading to stage redundancy.”
“Final Score: 3”

Input:
-	Image A: The first uploaded photo.
-	Image B: The second uploaded photo.
-	Instruction: {Instruction}

Output Format:
After completing the evaluation, please output the result in the following format(X is required to be an integer rating from 1 to 5):

Final Score: X
"""

prompt_scientific_plausibility = """You are an image process evaluation expert with profound knowledge literacy, particularly skilled at accurately judging the rationality and correctness of process images based on real processes (such as underlying mechanisms, scientific principles, chemical reactions, key features, etc.). Please conduct a strict evaluation of the input Image B. You will receive the following inputs:
-	Image A: Image A consists of two parts. On the left side (or top) is the reference start image, and on the right side (or bottom) is the reference end image.
-	Image B: The "intermediate transition process" image generated based on the reference start and end images.
-	Instruction: A description of how to transition from the reference start image to the reference end image to generate Image B, requiring Image B to fully reflect the intermediate process in grid format.
-	Checklist: Compiled from scientific knowledge or key process features, listing point by point the details and elements that the intermediate process should cover.

Your task:
Evaluate, item by item, whether the content in Image B correctly expresses the key features listed in the checklist.

Scoring criteria (maximum score is 5):
To prevent lenient evaluations or default assumptions that the generated result is reasonable, please use strict standards to examine whether Image B has any deficiencies, omissions, or unclear expressions, and reflect these in your scoring. Do not assign high scores simply because of overall stylistic harmony or subjective speculation about intent. Scoring must be determined using the strictest and most conservative standards.
-	5 (Perfectly aligned): Image B perfectly presents all checklist items.
-	4 (Well aligned): Image B presents all checklist items well, with only minor deviations.
-	3 (Generally aligned): Image B presents all checklist items, though deviations exist, it still reasonably reflects the checklist.
-	2 (Largely misaligned): Image B does not present all checklist items, with missing elements and poor overall rationality.
-	1 (Completely misaligned): Image B fails entirely to meet the checklist requirements, losing overall rationality.

Evaluation guidance:
-	If Image B merely replicates the start and end states provided in Image A without focusing on the intermediate process, then Image B does not meet the basic requirement of expressing the intermediate transition process.
-	If Image B expresses the intermediate transition process, analyze the explicitly presented objective content of Image B based on the checklist and its descriptions, and evaluate how well Image B aligns with the checklist items.

Input:
-	Image B: The first uploaded photo.
-	Instruction: {Instruction}
-	Checklist: {Checklist}

Output format:
After completing the evaluation, please output the result as follows(X is required to be an integer rating from 1 to 5):

Final Score: X
"""


prompt_process_plausibility = """You are an image content analysis expert. Based on the following inputs, evaluate whether the model truly understands the “intermediate transition process from the reference start image to the reference end image.” You will receive the following inputs:
-	Instruction 1: Describe how to transition from the reference start image to the reference end image to generate Image B, including explicit intermediate transition path constraints.
-	Instruction 2: Describe how to transition from the reference start image to the reference end image to generate Image C, including explicit intermediate transition path constraints.
-	Image A: Composed of two parts—left/top as the reference start image, right/bottom as the reference end image.
-	Image B: The intermediate transition process result generated from Image A’s start/end images (should comply with the path constraints in Instruction 1).
-	Image C: The intermediate transition process result generated from Image A’s start/end images (should comply with the path constraints in Instruction 2).

Evaluation Task:
Determine whether the model truly understands and clearly expresses the intermediate transition process from start to end, strictly follows the path constraints in Instruction 1 and Instruction 2 respectively, and reflects differentiation between the two paths.

Scoring Criteria (Maximum 5 points):
Do not relax the standard due to overall stylistic harmony or subjective speculation of intent; score only based on explicitly presented content in Images B and C. Please do not assign a higher score simply because the overall style appears coordinated or reasonable. Use the strictest and most conservative standard for judgment.
-	5 points (Complete Understanding): Both B and C accurately, clearly, and with high quality reproduce the full transition process, strictly conforming to their respective path constraints; demonstrates strong understanding and differentiation ability.
-	4 points (Good Understanding): B and C reflect the transition process well, meet the corresponding path constraints, and show generally good understanding.
-	3 points (Average Understanding): B and C roughly present the transition process, basically reflect the path constraints, but contain inaccuracies.
-	2 points (Poor Understanding): B and C show transitions but lack clear path differentiation or fail to fully implement the constraints; unable to generate according to the required paths.
-	1 point (No Understanding): B and C cannot reasonably reflect the intermediate process, paths are invalid/chaotic, do not match the textual instructions.

Key Evaluation Points (Check item by item):
-	Explicitness and completeness of intermediate process: (1) Do B and C clearly show “intermediate steps,” rather than simply copying or slightly modifying the start/end states? (2) Steps must be presented sequentially in a grid format (each grid as one stage, with the stage number in the top-left corner); do not rely on common sense or assumed knowledge to fill in unexpressed steps.
-	Conformance to path constraints (verify item by item): (1) In B and C, does each step explicitly correspond to the path constraints described in their respective instructions (explicit evidence only)? (2) “Looks reasonable overall” cannot substitute for explicit compliance.
-	Path understanding and differentiation ability: (1) Under different path constraints, do B and C show distinct intermediate processes and stage sequences? (2) Check for skipped stages, redundant stages, or missing stages, and deduct points accordingly.

Examples:
-	“B explicitly shows the intermediate process path, but deviates somewhat from the path requirements; C’s final result fits, but intermediate steps contain stage skipping/redundancy, failing to reflect the complete path process.”
-	“Final score: 2”

Input:
-	Instruction 1: {Instruction_A}
-	Instruction 2: {Instruction_B}
-	Image A: First uploaded photo.
-	Image B: Second uploaded photo.
-	Image C: Third uploaded photo.

Output format:
After completing the evaluation, please output the result in the following format:

Final score: X
"""