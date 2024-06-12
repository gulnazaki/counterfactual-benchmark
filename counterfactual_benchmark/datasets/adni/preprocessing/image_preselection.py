import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", '-c', type=str, help="Path to csv file 'ADNI1_Complete_1Yr_1.5T_<download_date>_.csv'")
    return parser.parse_args()


args = parse_arguments()
df15t = pd.read_csv(args.csv)
print("Shape of 1.5T data", df15t.shape)
print(df15t.head())

print('Before filtering')
print('Number of 1.5T images:', df15t['Image Data ID'].nunique())
print('Number of 1.5T subjects:', df15t.Subject.nunique())

raw_data_dir = './raw_data'

print('\nAfter filtering')
print('Number of 1.5T images:', df15t['Image Data ID'].nunique())
print('Number of 1.5T subjects:', df15t.Subject.nunique())

cols_to_drop = ['Modality', 'Downloaded', 'Type', 'Format', 'Sex', 'Age']
df15t.drop(columns=cols_to_drop, inplace=True)

print("Shape of 1.5T data", df15t.shape)

df15t['T'] = '1.5T'
df = df15t.sort_values(['Subject', 'Acq Date'])
df['Acq Date'] = pd.to_datetime(df['Acq Date'])

print('Number of images:', df['Image Data ID'].nunique())
print('Number of subjects:', df.Subject.nunique())
print('Shape: ', df.shape)

# Drop the images with descriptions ending with '2' as well as non 'MPR' or 'MPR-R' ones
descripions_to_drop_mask = df['Description'].str.endswith('2') | (~df['Description'].str.startswith('MPR'))
print('Number of images dropped: ', descripions_to_drop_mask.sum())
df = df[~descripions_to_drop_mask]
print('Shape:', df.shape)
print('Number of images:', df['Image Data ID'].nunique())
print('Number of subjects:', df.Subject.nunique())

# Number of images per subject by magnetic field strength (T)
image_counts_per_subject_by_T = df.pivot_table(values='Image Data ID', index='Subject', aggfunc='count', columns='T')
image_counts_per_subject_by_T.describe().convert_dtypes()

# Number of images per subject by visit
image_counts_per_subject_by_visit = df.pivot_table(
    values='Image Data ID',
    index='Subject',
    aggfunc='count',
    columns='Visit',
    fill_value=0
)
image_counts_per_subject_by_visit.describe().convert_dtypes()

num_duplicated_visits = df.duplicated(['Subject', 'Visit']).sum()
print(f'There are {num_duplicated_visits} total duplicate combinations of subjects and visits')
num_duplicated_visits = df.duplicated(['Subject', 'Acq Date']).sum()
print(f'There are {num_duplicated_visits} total duplicate combinations of subjects and dates')

# Sort the duplicated visit images by description (longer description goes to the bottom)
df = df.sort_values(['Subject', 'Visit', 'Acq Date', 'Description'])
# Set the index for a clearer view of the hierarchy
df[df.duplicated(['Subject', 'Visit', 'Acq Date'], keep=False)]\
  .set_index(["Subject", "Visit", "Acq Date", "Group"])\
  .head(100)

# Drop duplicated images keeping the last one (the longer description after sorting)
df = df.drop_duplicates(["Subject", "Visit", 'Acq Date'], keep='last')
df.set_index(["Subject", "Visit", "Acq Date", "Group"]).head(30)

df.pivot_table(
    values='Image Data ID',
    index='Subject',
    aggfunc='count',
    columns='Acq Date',
    fill_value=0
).max().max()

print('Shape:', df.shape)
print('Number of images:', df['Image Data ID'].nunique())
print('Number of subjects:', df.Subject.nunique())

# Add the group column
subject_group_pivot = df.pivot_table(values='Image Data ID', index='Subject', columns='Group', aggfunc='count')
image_counts_per_subject = image_counts_per_subject_by_visit.apply('sum', axis=1).rename('Image Count').to_frame()
image_counts_per_subject = image_counts_per_subject.merge(
    subject_group_pivot.idxmax(axis=1).rename('Group'),
    right_index=True,
    left_index=True
)
subject_count_per_image_count_by_group = image_counts_per_subject\
    .reset_index()\
    .pivot_table(
        index='Image Count',
        values='Subject',
        columns='Group',
        aggfunc='count',
        fill_value=0
    )
subject_count_per_image_count_by_group

# Save the subject diagnosis labels for later use
subject_group_pivot.idxmax(axis=1).rename('Group').value_counts().to_csv('subject_groups.csv')
# Save the images ids
df['Image Data ID'].to_csv('final_image_ids.csv', index=False)
