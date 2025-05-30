# Analysis

```python
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")
```

```python
# Display first few rows of the dataframe
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Size of the data
data.size
```
    232584

```python
# Shape of the data
data.shape
```
    (19382, 12)

```python
# Basic information about the data
data.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB

```python
# Descriptive statistics
data.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>


### Visualizations

#### **video_duration_sec**

All videos range from 5 to 60 seconds in length, with a uniform distribution.

```python
# Boxplot to visualize distribution of `video_duration_sec`
plt.figure(figsize=(5, 1))
sns.boxplot(x=data['video_duration_sec'])
plt.title('Video Duration')
plt.xlabel('Seconds')
plt.show()
```

![output_32_0](https://github.com/user-attachments/assets/877efb68-a9bd-43a6-b179-f2a9ec039050)


```python
# Histogram to visualize distribution of `video_duration_sec`

plt.figure()
sns.histplot(data['video_duration_sec'], bins=range(0,61,5))
plt.title('Video Duration')
plt.xlabel('Seconds')
plt.ylabel('Videos')
plt.tight_layout()
plt.show()
```

![output_34_0](https://github.com/user-attachments/assets/28a6164e-92d3-4794-be93-a45d3d214313)


#### **video_view_count**

Video view count has a highly uneven distribution, with over half of the videos receiving fewer than 100,000 views. For view counts above 100,000, the distribution is uniform.

```python
# Boxplot to visualize distribution of `video_view_count`

plt.figure(figsize=(5, 1))
sns.boxplot(x=data['video_view_count'])
plt.title('Video Views')
plt.xlabel('Views')
plt.show()
```

![output_37_0](https://github.com/user-attachments/assets/40c3b6aa-40fc-4c01-91c8-96905ebaf952)


```python
# Histogram to visualize distribution of `video_view_count`

plt.figure()
sns.histplot(data['video_view_count'], bins='auto')
plt.title('Video Views')
plt.xlabel('Views')
plt.ylabel('Videos')
plt.tight_layout()
plt.show()
```

![output_39_0](https://github.com/user-attachments/assets/9225ecb7-bc5e-47a9-9e6d-0fdb9ddf5abb)


#### **video_like_count**

Similar to view count, significantly more videos have fewer than 100,000 likes than those with more. However, the distribution tapers off more gradually in this case, with a right skew and many videos clustered at the higher end of the like count.

```python
# Boxplot to visualize distribution of `video_like_count`

plt.figure(figsize=(10, 1))
sns.boxplot(x=data['video_like_count'])
plt.title('Video Likes')
plt.xlabel('Likes')
plt.show()
```

![output_42_0](https://github.com/user-attachments/assets/6c5c0c06-50ee-4ca4-8a59-61b971c5ce47)


```python
# Histogram to visualize distribution of `video_like_count`

bins = range(0, 701_000, 100_000)
labels = ['0'] + [f'{i}k' for i in range(100, 701, 100)]

plt.figure()
ax = sns.histplot(data['video_like_count'], bins=bins)
ax.set_xticks(bins)
ax.set_xticklabels(labels)
ax.set(title='Video Likes', xlabel='Likes', ylabel='Videos')
plt.tight_layout()
plt.show()
```

![output_44_0](https://github.com/user-attachments/assets/fa1b55d2-2fb8-4185-8c87-d0a874f76d7c)


#### **video_comment_count**

Once again, the vast majority of videos fall at the lower end of the comment count range, with most receiving fewer than 100 comments. The distribution is heavily right-skewed.

```python
# Boxplot to visualize distribution of `video_comment_count`

plt.figure(figsize=(10, 1))
sns.boxplot(x=data['video_comment_count'])
plt.title('Video Comments')
plt.xlabel('Comments')
plt.show()
```

![output_47_0](https://github.com/user-attachments/assets/927f949d-2a11-476f-8f18-05c11beaf92c)


```python
# Histogram to visualize distribution of `video_comment_count`

plt.figure()
sns.histplot(data['video_comment_count'], bins=range(0,(3001),100))
plt.title('Video Comments')
plt.xlabel('Comments')
plt.ylabel('Videos')
plt.tight_layout()
plt.show()
```

![output_49_0](https://github.com/user-attachments/assets/9c5e7b73-8f3b-4b35-adca-6c364754ccb3)


#### **video_share_count**

The overwhelming majority of videos received fewer than 10,000 shares, with the distribution being heavily right-skewed.

```python
# Boxplot to visualize distribution of `video_share_count`

plt.figure(figsize=(10, 1))
sns.boxplot(x=data['video_share_count'])
plt.title('Video Shares')
plt.xlabel('Shares')
plt.show()
```

![output_52_0](https://github.com/user-attachments/assets/232abe4f-5d10-4fc2-be81-237b438e91b6)


```python
# Histogram to visualize distribution of `video_share_count`

plt.figure()
sns.histplot(data['video_share_count'], bins=range(0,(3001),100))
plt.title('Video Shares')
plt.xlabel('Shares')
plt.ylabel('Videos')
plt.tight_layout()
plt.show()
```

![output_54_0](https://github.com/user-attachments/assets/2f0c036a-c9e3-4659-a283-adb3ac72a935)


#### **video_download_count**

The majority of videos were downloaded fewer than 500 times, though some exceeded 12,000 downloads. Once again, the distribution is heavily right-skewed.

```python
# Boxplot to visualize distribution of `video_download_count`

plt.figure(figsize=(10, 1))
sns.boxplot(x=data['video_download_count'])
plt.title('Video Downloads')
plt.xlabel('Downloads')
plt.show()
```

![output_57_0](https://github.com/user-attachments/assets/d00095a6-5442-4a72-9070-249c211a779b)

```python
# Histogram to visualize distribution of `video_download_count`

plt.figure()
sns.histplot(data['video_download_count'], bins=range(0,(15001),500))
plt.title('Video Downloads')
plt.xlabel('Downloads')
plt.ylabel('Videos')
plt.tight_layout()
plt.show()
```

![output_59_0](https://github.com/user-attachments/assets/82793674-98fc-48bf-91e7-bdf8308d4e9a)


#### Claim status by verification status

Verified users are far fewer than unverified ones, but they are significantly more likely to post opinions.

```python
# Histogram to visualize claim status and verification status

plt.figure(figsize=(7,4))
sns.histplot(data=data,
             x='claim_status',
             hue='verified_status',
             multiple='dodge',
             shrink=0.9)
plt.title('Claims by verification status histogram');
```

![output_62_0](https://github.com/user-attachments/assets/dcb0e8e0-9b9b-4fab-b11c-274afa028609)


#### **Claim status by author ban status**

For both claims and opinions, active authors greatly outnumber those who are banned or under review. However, the proportion of active authors is much higher for opinion videos than for claim videos, suggesting that authors of claim videos are more likely to be reviewed or banned.

```python
# Histogram to visualize claim status by author ban status

fig = plt.figure(figsize=(7,4))
sns.histplot(data, x='claim_status', hue='author_ban_status',
             multiple='dodge',
             hue_order=['active', 'under review', 'banned'],
             shrink=0.9,
             palette={'active':'green', 'under review':'orange', 'banned':'red'},
             alpha=0.5)
plt.title('Claim Status by Author Ban Status');
```

![output_65_0](https://github.com/user-attachments/assets/9d690607-60ab-4d9e-828e-62287dd1b8b0)


#### **Median view counts by ban status**

The median view counts for non-active authors are significantly higher than those for active authors. Given that non-active authors are more likely to post claims, and their videos receive substantially more views overall, video_view_count could serve as a useful indicator of claim status.

```python
# Bar plot to visualize median views by ban status

sns.barplot(data=data,
            x='author_ban_status', y='video_view_count',
            estimator=np.median, ci=None,
            order=['active', 'under review', 'banned'],
            palette={'active':'green', 'under review':'orange', 'banned':'red'})

plt.title('Median Views per Ban Status')
plt.xlabel('Ban Status')
plt.ylabel('Median Views')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

![output_68_0](https://github.com/user-attachments/assets/0c1d79cc-6cc4-4499-9d4b-07bf75711bee)


```python
# Median view count for claim status

data.groupby('claim_status')['video_view_count'].median()
```

    claim_status
    claim      501555.0
    opinion      4953.0
    Name: video_view_count, dtype: float64


#### Total views by claim status

Although the dataset contains roughly equal numbers of claim and opinion videos, the overall view count is heavily dominated by claim videos.

```python
# Pie chart showing the porportion of views by video claim status

fig = plt.figure(figsize=(3,3))
plt.pie(data.groupby('claim_status')['video_view_count'].sum(), labels=['claim', 'opinion'])
plt.title('Total views by video claim status');
```

![output_72_0](https://github.com/user-attachments/assets/c510c5ed-a470-4180-945f-1c75f51491ea)


### Outliers

    Outliers for video_view_count: 2343
    Outliers for video_like_count: 3468
    Outliers for video_share_count: 3732
    Outliers for video_download_count: 3733
    Outliers for video_comment_count: 3882
    
```python
# Number of outliers (median + 1.5 * IQR)

count_cols = ['video_view_count',
              'video_like_count',
              'video_share_count',
              'video_download_count',
              'video_comment_count']

for col in count_cols:
    median = data[col].median()
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    threshold = median + 1.5 * iqr
    outliers = data[data[col] > threshold]
    count = len(outliers)
    
    print(f"Outliers for {col}: {count}")
```


#### Scatterplots


```python
# Scatterplot of `video_view_count` versus `video_like_count` according to 'claim_status'

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='video_view_count', y='video_like_count', hue='claim_status', s=5)

plt.title('Views vs. Likes for Claim Status')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

![output_77_0](https://github.com/user-attachments/assets/e6a2225e-0c64-470f-a278-289275ce55b6)


```python
# Scatterplot of `video_view_count` versus `video_like_count` for opinions only

filtered_df = data[data['claim_status'] == 'opinion']

plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_df, x='video_view_count', y='video_like_count', s=5)

plt.title('Views vs. Likes for Opinions')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

![output_78_0](https://github.com/user-attachments/assets/572f7bbb-51e3-463e-babb-d32e279a9d32)
