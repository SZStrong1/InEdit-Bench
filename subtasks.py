import pandas as pd
import numpy as np


def subtasks_scores(input_file, output_file):
    # ============================================================================================================
    # dynamic_process
    # ============================================================================================================
    biology_nature_indices = ['dynamic_process_1', 'dynamic_process_2', 'dynamic_process_6', 'dynamic_process_7',
                              'dynamic_process_9',
                              'dynamic_process_11', 'dynamic_process_12', 'dynamic_process_30', 'dynamic_process_44',
                              'dynamic_process_48',
                              'dynamic_process_53', 'dynamic_process_59', 'dynamic_process_61']
    coordinated_motion_indices = ['dynamic_process_8', 'dynamic_process_21', 'dynamic_process_22', 'dynamic_process_31',
                                  'dynamic_process_33',
                                  'dynamic_process_36', 'dynamic_process_38', 'dynamic_process_39',
                                  'dynamic_process_40', 'dynamic_process_45',
                                  'dynamic_process_52', 'dynamic_process_57', 'dynamic_process_58']
    daily_life_indices = ['dynamic_process_3', 'dynamic_process_5', 'dynamic_process_10', 'dynamic_process_13',
                          'dynamic_process_14',
                          'dynamic_process_17', 'dynamic_process_18', 'dynamic_process_19', 'dynamic_process_23',
                          'dynamic_process_24',
                          'dynamic_process_25', 'dynamic_process_26', 'dynamic_process_27', 'dynamic_process_28',
                          'dynamic_process_29',
                          'dynamic_process_46', 'dynamic_process_49', 'dynamic_process_54', 'dynamic_process_60',
                          'dynamic_process_62',
                          'dynamic_process_63']
    mechanical_operations_indices = ['dynamic_process_4', 'dynamic_process_15', 'dynamic_process_37',
                                     'dynamic_process_41', 'dynamic_process_47',
                                     'dynamic_process_50', 'dynamic_process_51', 'dynamic_process_55',
                                     'dynamic_process_56']
    sudden_events_indices = ['dynamic_process_16', 'dynamic_process_20', 'dynamic_process_32', 'dynamic_process_34',
                             'dynamic_process_35',
                             'dynamic_process_42', 'dynamic_process_43', 'dynamic_process_64', 'dynamic_process_65']

    # ============================================================================================================
    # scientific_simulation
    # ============================================================================================================
    biology_indices = ['scientific_simulation_7', 'scientific_simulation_17', 'scientific_simulation_18',
                       'scientific_simulation_20', 'scientific_simulation_21',
                       'scientific_simulation_22', 'scientific_simulation_23']
    chemistry_indices = ['scientific_simulation_2', 'scientific_simulation_3', 'scientific_simulation_10',
                         'scientific_simulation_11', 'scientific_simulation_14',
                         'scientific_simulation_15', 'scientific_simulation_19']
    physics_indices = ['scientific_simulation_1', 'scientific_simulation_4', 'scientific_simulation_5',
                       'scientific_simulation_6', 'scientific_simulation_8',
                       'scientific_simulation_9', 'scientific_simulation_12', 'scientific_simulation_13',
                       'scientific_simulation_16']

    # ============================================================================================================
    # state_transition
    # ============================================================================================================
    construction_assembly_indices = ['state_transition_2', 'state_transition_3', 'state_transition_4',
                                     'state_transition_5', 'state_transition_11', 'state_transition_14',
                                     'state_transition_24', 'state_transition_30', 'state_transition_32',
                                     'state_transition_33', 'state_transition_34', 'state_transition_35',
                                     'state_transition_43', 'state_transition_48']
    decoration_painting_indices = ['state_transition_1', 'state_transition_6', 'state_transition_9',
                                   'state_transition_13', 'state_transition_23', 'state_transition_26',
                                   'state_transition_28', 'state_transition_36', 'state_transition_37',
                                   'state_transition_38', 'state_transition_39', 'state_transition_41']
    organization_arrangement_indices = ['state_transition_10', 'state_transition_12', 'state_transition_15',
                                        'state_transition_16', 'state_transition_20',
                                        'state_transition_21', 'state_transition_27', 'state_transition_29',
                                        'state_transition_40', 'state_transition_42']
    processing_deformation_indices = ['state_transition_7', 'state_transition_8', 'state_transition_17',
                                      'state_transition_18', 'state_transition_19', 'state_transition_22',
                                      'state_transition_25', 'state_transition_31', 'state_transition_44',
                                      'state_transition_45', 'state_transition_46', 'state_transition_47',
                                      'state_transition_49']

    # ============================================================================================================
    # temporal_sequence
    # ============================================================================================================
    environment_society_indices = ['temporal_sequence_13', 'temporal_sequence_14', 'temporal_sequence_15',
                                   'temporal_sequence_17', 'temporal_sequence_18', 'temporal_sequence_19',
                                   'temporal_sequence_20', 'temporal_sequence_21', 'temporal_sequence_23',
                                   'temporal_sequence_31', 'temporal_sequence_32', 'temporal_sequence_42',
                                   'temporal_sequence_43', 'temporal_sequence_44', 'temporal_sequence_51',
                                   'temporal_sequence_54', 'temporal_sequence_57', 'temporal_sequence_58',
                                   'temporal_sequence_59']

    growth_decay_indices = ['temporal_sequence_3', 'temporal_sequence_9', 'temporal_sequence_10',
                            'temporal_sequence_16', 'temporal_sequence_27', 'temporal_sequence_28',
                            'temporal_sequence_38',
                            'temporal_sequence_48', 'temporal_sequence_49', 'temporal_sequence_53',
                            'temporal_sequence_56', 'temporal_sequence_61', 'temporal_sequence_62',
                            'temporal_sequence_63',
                            'temporal_sequence_66']

    physical_transformation_indices = ['temporal_sequence_1', 'temporal_sequence_2', 'temporal_sequence_6',
                                       'temporal_sequence_12', 'temporal_sequence_22', 'temporal_sequence_24',
                                       'temporal_sequence_25',
                                       'temporal_sequence_26', 'temporal_sequence_29', 'temporal_sequence_30',
                                       'temporal_sequence_33', 'temporal_sequence_34', 'temporal_sequence_35',
                                       'temporal_sequence_36',
                                       'temporal_sequence_37', 'temporal_sequence_39', 'temporal_sequence_40',
                                       'temporal_sequence_41', 'temporal_sequence_45', 'temporal_sequence_46',
                                       'temporal_sequence_50',
                                       'temporal_sequence_52', 'temporal_sequence_55', 'temporal_sequence_64',
                                       'temporal_sequence_65']

    temporal_measurement_indices = ['temporal_sequence_4', 'temporal_sequence_5', 'temporal_sequence_7',
                                    'temporal_sequence_8', 'temporal_sequence_11', 'temporal_sequence_47',
                                    'temporal_sequence_60']

    # ===========================================================================================================
    # ===========================================================================================================
    indices_list = [
        biology_nature_indices, coordinated_motion_indices, daily_life_indices,
        mechanical_operations_indices, sudden_events_indices,
        biology_indices, chemistry_indices, physics_indices,
        construction_assembly_indices, decoration_painting_indices,
        organization_arrangement_indices, processing_deformation_indices,
        environment_society_indices, growth_decay_indices,
        physical_transformation_indices, temporal_measurement_indices
    ]

    excel_indices = [
        "biology_nature_indices", "coordinated_motion_indices", "daily_life_indices",
        "mechanical_operations_indices", "sudden_events_indices", "biology_indices",
        "chemistry_indices", "physics_indices", "construction_assembly_indices",
        "decoration_painting_indices", "organization_arrangement_indices",
        "processing_deformation_indices", "environment_society_indices",
        "growth_decay_indices", "physical_transformation_indices", "temporal_measurement_indices"
    ]
    # ===========================================================================================================
    # ===========================================================================================================
    def accuracy_average_scores(df, target_indices):
        filtered_df = df[df["index"].isin(target_indices)].copy()
        category_index = target_indices[0].split('_')[0]
        if category_index in ['dynamic', 'scientific']:
            score_cols = [
                "scores_appearance_consistency_all",
                "scores_perceptual_quality_all",
                "scores_semantic_consistency_all",
                "scores_logical_coherence_all",
                "scores_scientific_plausibility_all"
            ]
        elif category_index in ["state", "temporal"]:
            score_cols = [
                "scores_appearance_consistency_all",
                "scores_perceptual_quality_all",
                "scores_semantic_consistency_all",
                "scores_logical_coherence_all"
            ]

        filtered_df[score_cols] = filtered_df[score_cols].apply(pd.to_numeric, errors='coerce')
        # accuracy
        perfect_rows = filtered_df[score_cols].apply(lambda row: (row == 5).all(), axis=1)
        perfect_indices = filtered_df.loc[perfect_rows, "index"].tolist()
        total = len(filtered_df)
        perfect_count = perfect_rows.sum()
        ratio = perfect_count / total if total > 0 else 0.0
        # average_scores
        filtered_df[score_cols] = filtered_df[score_cols].replace(0, np.nan)
        average_scores = filtered_df[score_cols].mean(skipna=True)
        average_scores = (average_scores - 1) * 25
        overall_average_score = average_scores.mean()
        return average_scores, overall_average_score, ratio, perfect_indices

    df = pd.read_excel(input_file, header=0)
    all_results = []

    for indices in indices_list:
        average_scores, overall_average_score, ratio, perfect_indices = accuracy_average_scores(df, indices)
        ratio = round(ratio * 100, 2)
        overall_average_score = round(overall_average_score, 2)
        appearance_consistency = round(average_scores.iloc[0], 2)
        perceptual_quality = round(average_scores.iloc[1], 2)
        semantic_consistency = round(average_scores.iloc[2], 2)
        logical_coherence = round(average_scores.iloc[3], 2)
        if indices[0].split('_')[0] in ['dynamic', 'scientific']:
            scientific_plausibility = round(average_scores.iloc[4], 2)
        else:
            scientific_plausibility = None

        all_results.append([appearance_consistency, perceptual_quality, semantic_consistency, logical_coherence,
                            scientific_plausibility, overall_average_score, ratio, perfect_indices])

    columns = [
        "appearance_consistency", "perceptual_quality", "semantic_consistency",
        "logical_coherence", "scientific_plausibility", "overall_average_score",
        "ratio", "perfect_indices"
    ]
    df_results = pd.DataFrame(all_results, columns=columns, index=excel_indices)
    df_results.to_excel(output_file, index=True)




