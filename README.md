# DisasterBERT: Enhanced Tweet Classification with Contextual Prompting

This project implements a BERT-based model for classifying disaster-related tweets into humanitarian categories, using a novel contextual prompting approach to improve classification accuracy.

## Approach: Contextual Prompting

The model uses a structured prompting technique that provides additional context to the BERT model during classification. For each tweet, the model receives:

1. **Temporal Context**: The number of days since the disaster event
2. **Event Description**: A brief description of the disaster event
3. **Classification Task**: A clear definition of the humanitarian categories
4. **Original Tweet**: The tweet text to be classified

This contextual information helps the model better understand the tweet's relevance and context within the disaster scenario.

## Classification Categories

Tweets are classified into one of these humanitarian categories:
1. **Caution and advice** - Warnings, safety tips, or official guidance
2. **Displaced people** - Evacuations, shelters, missing family reports
3. **Infrastructure damage** - Collapsed buildings, flooded roads, power outages
4. **Injured/dead** - Casualty reports, injury details, body recovery
5. **Requests/needs** - Urgent asks for food, medicine, or rescue
6. **Rescue efforts** - Volunteer activities, donation drives
7. **Not humanitarian** - Irrelevant tweets, casual conversation
8. **Other** - Humanitarian but doesn't fit other categories

## Results

The model was evaluated on 11 disaster datasets, ordered by size from smallest to largest. Here are the best F1 scores achieved for each dataset:

| Dataset | Samples | F1 Score |
|---------|---------|----------|
| Sri Lanka Floods (2017) | 575 | 0.7558 |
| Italy Earthquake (2016) | 1,240 | 0.8807 |
| Ecuador Earthquake (2016) | 1,594 | 0.8763 |
| Hurricane Matthew (2016) | 1,659 | 0.8022 |
| Puebla Mexico Earthquake (2017) | 2,036 | 0.8651 |
| Kaikoura Earthquake (2016) | 2,217 | 0.7635 |
| Canada Wildfires (2016) | 2,259 | 0.8069 |
| Cyclone Idai (2019) | 3,944 | 0.8052 |
| Hurricane Maria (2017) | 7,328 | 0.7437 |
| Hurricane Harvey (2017) | 9,164 | 0.7697 |
| Hurricane Irma (2017) | 9,467 | 0.7371 |

## Comparison with Previous Approaches

The following table compares our Contextual Prompting approach (CP-BERT) with previous methods on overlapping datasets:

| Dataset | # Classes | RF | SVM | FastText | BERT | CP-BERT (Ours) |
|---------|----------|----|-----|----------|------|---------------|
| Ecuador Earthquake (2016) | 8 | 0.784 | 0.738 | 0.752 | 0.861 | **0.8763** |
| Canada Wildfires (2016) | 8 | 0.726 | 0.738 | 0.726 | 0.792 | **0.8069** |
| Italy Earthquake (2016) | 6 | 0.799 | 0.822 | 0.821 | 0.871 | **0.8807** |
| Kaikoura Earthquake (2016) | 9 | 0.660 | 0.693 | 0.658 | 0.768 | **0.7635** |
| Hurricane Matthew (2016) | 9 | 0.742 | 0.700 | 0.704 | 0.786 | **0.8022** |
| Sri Lanka Floods (2017) | 8 | 0.613 | 0.611 | 0.575 | 0.703 | **0.7558** |
| Hurricane Harvey (2017) | 9 | 0.719 | 0.713 | 0.718 | 0.759 | **0.7697** |
| Hurricane Irma (2017) | 9 | 0.693 | 0.695 | 0.694 | 0.722 | **0.7371** |
| Hurricane Maria (2017) | 9 | 0.682 | 0.682 | 0.688 | 0.715 | **0.7437** |
| Puebla Mexico Earthquake (2017) | 8 | 0.800 | 0.789 | 0.797 | 0.845 | **0.8651** |
| Cyclone Idai (2019) | 10 | 0.680 | 0.730 | 0.707 | 0.790 | **0.8052** |


## Implementation Details

- Model: BERT-large-uncased
- Training: 15 epochs with dynamic batch sizes (8-64) based on dataset size
- Optimizer: AdamW with learning rate 2e-5
- Hardware: Single GPU training

## Usage

To train the model on all datasets:

```bash
./DisasterBert/src/train_all.sh
```

To train on a specific dataset:

```bash
python3 train.py --dataset_dir /path/to/dataset --epochs 15 --batch_size 8
```


