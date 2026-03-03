import requests
import json
import argparse
import os
import os.path as osp
from typing import Callable, Iterable, Sized
import time
from utils import *
import pandas as pd
from subtasks import subtasks_scores


api_base = ''
api_key = ''
def gpt_generate(inputs, model='gpt-4o-2024-11-20', temperature=0, max_tokens=10000, **kwargs):
    input_msgs = prepare_inputs(inputs)
    temperature = kwargs.pop('temperature', temperature)
    max_tokens = kwargs.pop('max_tokens', max_tokens)
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    payload = dict(
        model=model,
        messages=input_msgs,
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
        **kwargs)
    response = requests.post(
        api_base,
        headers=headers, data=json.dumps(payload), timeout=60)
    ret_code = response.status_code
    ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
    answer = 'Failed to obtain answer via API. '
    try:
        resp_struct = json.loads(response.text)
        answer = resp_struct['choices'][0]['message']['content'].strip()
    except Exception as err:
        print(f'{type(err)}: {err}')
        print(response.text if hasattr(response, 'text') else response)

    return ret_code, answer, response

def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        pbar = tqdm(total=len(unfinished))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
                    if keys is not None:
                        res[keys[idx]] = results[idx]
            if len(new_finished):
                if save is not None:
                    dump(res, save)
                pbar.update(len(new_finished))
                for k in new_finished:
                    unfinished.remove(k)
            time.sleep(0.1)
        pbar.close()

    if save is not None:
        dump(res, save)
    return results


def eval_vanilla(item, input_dir, output_dir, **kwargs):
    category = item['category']
    if category == 'process_plausibility':
        index = item['index']
        instruct_A = item['instruction_A']
        instruct_B = item['instruction_B']
        output_dir = osp.join(output_dir, f'images/{category}')
        output_img_A = find_image(output_dir, index + "_A")
        output_img_B = find_image(output_dir, index + "_B")

        img_init = osp.join(input_dir, item['image'])
        prompt_process_plausibility_evalution = prompt_process_plausibility.format(Instruction_A=instruct_A,
                                                                               Instruction_B=instruct_B)
        message_process_plausibility_evalution = []
        text_prompt_process_plausibility_evalution = {'type': 'text', 'value': prompt_process_plausibility_evalution}
        image_init = {
            'type': 'image',
            'value': img_init,
        }
        image_output_A = {
            'type': 'image',
            'value': output_img_A,
        }
        image_output_B = {
            'type': 'image',
            'value': output_img_B,
        }
        #  message_process_plausibility_evalution
        message_process_plausibility_evalution.append(text_prompt_process_plausibility_evalution)
        message_process_plausibility_evalution.append(image_init)
        message_process_plausibility_evalution.append(image_output_A)
        message_process_plausibility_evalution.append(image_output_B)

        ret_code_process_plausibility, judge_process_plausibility, response_process_plausibility = gpt_generate(
            message_process_plausibility_evalution,
            **kwargs)
        return dict(judge_process_plausibility=judge_process_plausibility)

    else:
        instruct = item['instruction']
        index = item['index']
        category = item['category']
        output_dir = osp.join(output_dir, f'images/{category}')
        output_img = find_image(output_dir, index)

        if category in ['state_transition', 'temporal_sequence']:
            img_init = osp.join(input_dir, item['image'])
            prompt_appearance_consistency_evalution = prompt_appearance_consistency.format(Instruction=instruct)
            prompt_perceptual_quality_evalution = prompt_perceptual_quality
            prompt_semantic_consistency_evalution = prompt_semantic_consistency.format(Instruction=instruct)
            prompt_logical_coherence_evalution = prompt_logical_coherence.format(Instruction=instruct)
        else:
            img_init = osp.join(input_dir, item['image'])
            prompt_appearance_consistency_evalution = prompt_appearance_consistency.format(Instruction=instruct)
            prompt_perceptual_quality_evalution = prompt_perceptual_quality
            prompt_semantic_consistency_evalution = prompt_semantic_consistency.format(Instruction=instruct)
            prompt_logical_coherence_evalution = prompt_logical_coherence.format(Instruction=instruct)
            prompt_scientific_plausibility_evalution = prompt_scientific_plausibility.format(Instruction=instruct,
                                                                         Checklist=item['checklist'])

        message_appearance_consistency_evalution = []
        message_perceptual_quality_evalution = []
        message_semantic_consistency_evalution = []
        message_logical_coherence_evalution = []
        message_scientific_plausibility_evalution = []

        text_prompt_appearance_consistency_evalution = {'type': 'text',
                                                        'value': prompt_appearance_consistency_evalution}
        text_prompt_perceptual_quality_evalution = {'type': 'text', 'value': prompt_perceptual_quality_evalution}
        text_prompt_semantic_consistency_evalution = {'type': 'text', 'value': prompt_semantic_consistency_evalution}
        text_prompt_logical_coherence_evalution = {'type': 'text', 'value': prompt_logical_coherence_evalution}
        image_init = {
            'type': 'image',
            'value': img_init,
        }
        image_output = {
            'type': 'image',
            'value': output_img,
        }
        #  message_appearance_consistency_evalution
        message_appearance_consistency_evalution.append(text_prompt_appearance_consistency_evalution)
        message_appearance_consistency_evalution.append(image_init)
        message_appearance_consistency_evalution.append(image_output)

        # message_perceptual_quality_evalution
        message_perceptual_quality_evalution.append(text_prompt_perceptual_quality_evalution)
        message_perceptual_quality_evalution.append(image_output)

        # message_semantic_consistency_evalution
        message_semantic_consistency_evalution.append(text_prompt_semantic_consistency_evalution)
        message_semantic_consistency_evalution.append(image_init)
        message_semantic_consistency_evalution.append(image_output)

        # message_logical_coherence_evalution
        message_logical_coherence_evalution.append(text_prompt_logical_coherence_evalution)
        message_logical_coherence_evalution.append(image_init)
        message_logical_coherence_evalution.append(image_output)

        # answer_genrate
        ret_code_appearance_consistency, judge_appearance_consistency, response_appearance_consistency = gpt_generate(
            message_appearance_consistency_evalution, **kwargs)
        ret_code_perceptual_quality, judge_perceptual_quality, response_perceptual_quality = gpt_generate(
            message_perceptual_quality_evalution, **kwargs)
        ret_code_semantic_consistency, judge_semantic_consistency, response_semantic_consistency = gpt_generate(
            message_semantic_consistency_evalution, **kwargs)
        ret_code_logical_coherence, judge_logical_coherence, response_logical_coherence = gpt_generate(
            message_logical_coherence_evalution, **kwargs)

        if category in ['dynamic_process', 'scientific_simulation']:
            # message_scientific_plausibility_evalution
            text_prompt_scientific_plausibility_evalution = {'type': 'text', 'value': prompt_scientific_plausibility_evalution}
            message_scientific_plausibility_evalution.append(text_prompt_scientific_plausibility_evalution)
            message_scientific_plausibility_evalution.append(image_init)
            message_scientific_plausibility_evalution.append(image_output)

            ret_code_scientific_plausibility, judge_scientific_plausibility, response_scientific_plausibility = gpt_generate(
                message_scientific_plausibility_evalution,
                **kwargs)
            return dict(judge_appearance_consistency=judge_appearance_consistency, judge_perceptual_quality=judge_perceptual_quality,
                        judge_semantic_consistency=judge_semantic_consistency, judge_logical_coherence=judge_logical_coherence,
                        judge_scientific_plausibility=judge_scientific_plausibility)
        else:
            return dict(judge_appearance_consistency=judge_appearance_consistency, judge_perceptual_quality=judge_perceptual_quality,
                        judge_semantic_consistency=judge_semantic_consistency, judge_logical_coherence=judge_logical_coherence)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/data.json', help='Json Path')
    parser.add_argument('--output', type=str, default='./outputs/gpt-image-1', help='Output Image Dir, outputs/MODEL_NAME')
    parser.add_argument('--input', type=str, default='./data', help='Init Image Dir')
    parser.add_argument('--prefix', type=str, default=None, help='output json prefix')
    parser.add_argument('--model', type=str, default=None, help='Model Name')
    parser.add_argument('--nproc', type=int, default=1, help='n processes for api')

    args = parser.parse_args()

    model_name = args.output.split('/')[-1] if args.model is None else args.model
    if not args.prefix:
        tmp_file = f"{args.output}/{model_name}.pkl"
        judge_res = f"{args.output}/{model_name}_judge_res.xlsx"
        score_file = f"{args.output}/{model_name}_scores.xlsx"
        subtasks_score_file = f"{args.output}/{model_name}_subtasks_score.xlsx"
    else:
        tmp_file = f"{args.output}/{args.prefix}_{model_name}.pkl"
        judge_res = f"{args.output}/{args.prefix}_{model_name}_judge_res.xlsx"
        score_file = f"{args.output}/{args.prefix}_{model_name}_scores.xlsx"
        subtasks_score_file = f"{args.output}/{args.prefix}_{model_name}_subtasks_score.xlsx"

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = pd.DataFrame(data)

    result = {}
    if osp.exists(tmp_file):
        result = load(tmp_file)

    items = []

    for i in range(len(data)):
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(item=x, input_dir=args.input, output_dir=args.output) for x in items]
    keys = [x['index'] for x in items]

    if len(tups):
        res = track_progress_rich(eval_vanilla, tups, nproc=args.nproc, chunksize=args.nproc, save=tmp_file, keys=keys)
        result = load(tmp_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v

    judges = [result[i] for i in data['index']]

    scores, judge_combine = [], []

    for judge in judges:
        if 'judge_scientific_plausibility' not in judge:
            if 'judge_appearance_consistency' not in judge:
                judge_combine.append('judge_process_plausibility\n\n' + judge['judge_process_plausibility'])
                scores_process_plausibility = extract(judge['judge_process_plausibility'])
                score = scores_process_plausibility
            else:
                judge_combine.append('judge_appearance_consistency\n\n' + judge['judge_appearance_consistency'] +
                                     '\n\njudge_perceptual_quality\n\n' + judge['judge_perceptual_quality'] +
                                     '\n\njudge_semantic_consistency\n\n' + judge['judge_semantic_consistency'] +
                                     '\n\njudge_logical_coherence\n\n' + judge['judge_logical_coherence']
                                     )
                scores_appearance_consistency = extract(judge['judge_appearance_consistency'])
                scores_perceptual_quality = extract(judge['judge_perceptual_quality'])
                scores_semantic_consistency = extract(judge['judge_semantic_consistency'])
                scores_logical_coherence = extract(judge['judge_logical_coherence'])

                if scores_appearance_consistency == None:
                    scores_appearance_consistency = [0]
                if scores_perceptual_quality == None:
                    scores_perceptual_quality = [0]
                if scores_semantic_consistency == None:
                    scores_semantic_consistency = [0]
                if scores_logical_coherence == None:
                    scores_logical_coherence = [0]

                score = scores_appearance_consistency + scores_perceptual_quality + scores_semantic_consistency + scores_logical_coherence

        else:
            judge_combine.append('judge_appearance_consistency\n\n' + judge['judge_appearance_consistency'] +
                                 '\n\njudge_perceptual_quality\n\n' + judge['judge_perceptual_quality'] +
                                 '\n\njudge_semantic_consistency\n\n' + judge['judge_semantic_consistency'] +
                                 '\n\njudge_logical_coherence\n\n' + judge['judge_logical_coherence'] +
                                 '\n\njudge_scientific_plausibility\n\n' + judge['judge_scientific_plausibility']
                                 )
            scores_appearance_consistency = extract(judge['judge_appearance_consistency'])
            scores_perceptual_quality = extract(judge['judge_perceptual_quality'])
            scores_semantic_consistency = extract(judge['judge_semantic_consistency'])
            scores_logical_coherence = extract(judge['judge_logical_coherence'])
            scores_scientific_plausibility = extract(judge['judge_scientific_plausibility'])
            if scores_appearance_consistency == None :
                scores_appearance_consistency = [0]
            if scores_perceptual_quality == None:
                scores_perceptual_quality = [0]
            if scores_semantic_consistency == None:
                scores_semantic_consistency = [0]
            if scores_logical_coherence == None:
                scores_logical_coherence = [0]
            if scores_scientific_plausibility == None:
                scores_scientific_plausibility = [0]
            score = scores_appearance_consistency + scores_perceptual_quality + scores_semantic_consistency + scores_logical_coherence +\
                    scores_scientific_plausibility
        scores.append(score)
    match_log = []
    scores_appearance_consistency_all = []
    scores_perceptual_quality_all = []
    scores_semantic_consistency_all = []
    scores_logical_coherence_all = []
    scores_scientific_plausibility_all = []
    scores_process_plausibility_all = []
    record_score = {}

    for score in scores:
        if score:
            match_log.append('succeed')
            if len(score) == 1:
                scores_appearance_consistency_all.append(None)
                scores_perceptual_quality_all.append(None)
                scores_semantic_consistency_all.append(None)
                scores_logical_coherence_all.append(None)
                scores_scientific_plausibility_all.append(None)
                scores_process_plausibility_all.append(score[0])
            elif len(score) == 4:
                scores_appearance_consistency_all.append(score[0])
                scores_perceptual_quality_all.append(score[1])
                scores_semantic_consistency_all.append(score[2])
                scores_logical_coherence_all.append(score[3])
                scores_scientific_plausibility_all.append(None)
                scores_process_plausibility_all.append(None)
            elif len(score) == 5:
                scores_appearance_consistency_all.append(score[0])
                scores_perceptual_quality_all.append(score[1])
                scores_semantic_consistency_all.append(score[2])
                scores_logical_coherence.append(score[3])
                scores_scientific_plausibility_all.append(score[4])
                scores_process_plausibility_all.append(None)
        else:
            match_log.append('failed')
            scores_appearance_consistency_all.append(None)
            scores_perceptual_quality_all.append(None)
            scores_semantic_consistency_all.append(None)
            scores_logical_coherence_all.append(None)
            scores_scientific_plausibility_all.append(None)
            scores_process_plausibility_all.append(None)

    record_score['index'] = data['index']
    record_score['scores_appearance_consistency_all'] = scores_appearance_consistency_all
    record_score['scores_perceptual_quality_all'] = scores_perceptual_quality_all
    record_score['scores_semantic_consistency_all'] = scores_semantic_consistency_all
    record_score['scores_logical_coherence_all'] = scores_logical_coherence_all
    record_score['scores_scientific_plausibility_all'] = scores_scientific_plausibility_all
    record_score['scores_process_plausibility_all'] = scores_process_plausibility_all
    record_score['match_log'] = match_log
    record_score['judge_combine'] = judge_combine

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in record_score.items()]))
    df.to_excel(judge_res, index=False, engine='xlsxwriter')

    eval_scores = pd.read_excel(judge_res, header=0)
    indicator_scores_100, overall_average = indiactor_average(eval_scores)
    ratio, perfect_indices = accuracy_overall(eval_scores)
    ci_lower, ci_upper = calculate_bootstrap_ci(judge_res)
    ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"

    record_indicator_score = {
        "model": [model_name],
        "appearance_consistency": [round(indicator_scores_100.iloc[0], 2)],
        "perceptual_quality": [round(indicator_scores_100.iloc[1], 2)],
        "semantic_consistency": [round(indicator_scores_100.iloc[2], 2)],
        "logical_coherence": [round(indicator_scores_100.iloc[3], 2)],
        "scientific_plausibility": [round(indicator_scores_100.iloc[4], 2)],
        "process_plausibility": [round(indicator_scores_100.iloc[5], 2)],
        "overall_average": [round(overall_average, 2)],
        'CI Range String': ci_str,
        "ration": [round(ratio, 2)],
        "perfect_indices": [', '.join(perfect_indices)]
    }
    df = pd.DataFrame(record_indicator_score)
    df.to_excel(score_file, index=False)

    # subtasks_scores(judge_res, subtasks_score_file)


if __name__ == '__main__':
    main()
