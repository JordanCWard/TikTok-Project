# **Data Exploration**


### Summary

Videos by banned authors and those under review generally attract more views, likes, and shares compared to videos by active (non-banned) authors. However, once a video is viewed, its engagement rate is primarily influenced by its claim status rather than the author's ban status.

Claim videos typically achieve higher view rates than opinion videos. Additionally, claim videos receive higher average likes, indicating they are generally more positively received. They also garner greater engagement through comments and shares compared to opinion videos.

Notably, for claim videos, banned authors experience slightly higher engagement rates in terms of likes per view and shares per view than active authors or authors under review. Conversely, for opinion videos, active authors and those under review show higher engagement rates across all categories compared to banned authors.

Out of the 19,382 videos in this dataset, nearly half (9,608) are classified as claim videos. Engagement levels are strongly correlated with claim status, making this an important area for further research.

Overall, videos by banned authors exhibit significantly higher engagement levels than those by active authors, with authors under review having engagement levels intermediate between these two groups.










<!--
Learn about the dataset by using:
head()
tail()
describe()
columns
dtypes
nunique()
isnull().sum()
value_counts()
sample(n)
corr()
hist()
plot(kind='box')
df['col'].value_counts(normalize=True)
-->

### Analysis

```python
# Import packages for loading and exploring the dataset.
import pandas as pd
import numpy as np

# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")

# Display first ten rows of the dataframe
data.head(10)
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
    <tr>
      <th>5</th>
      <td>6</td>
      <td>claim</td>
      <td>8972200955</td>
      <td>35</td>
      <td>someone shared with me that gross domestic pro...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>336647.0</td>
      <td>175546.0</td>
      <td>62303.0</td>
      <td>4293.0</td>
      <td>1857.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>claim</td>
      <td>4958886992</td>
      <td>16</td>
      <td>someone shared with me that elvis presley has ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>750345.0</td>
      <td>486192.0</td>
      <td>193911.0</td>
      <td>8616.0</td>
      <td>5446.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>claim</td>
      <td>2270982263</td>
      <td>41</td>
      <td>someone shared with me that the best selling s...</td>
      <td>not verified</td>
      <td>active</td>
      <td>547532.0</td>
      <td>1072.0</td>
      <td>50.0</td>
      <td>22.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>claim</td>
      <td>5235769692</td>
      <td>50</td>
      <td>someone shared with me that about half of the ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>24819.0</td>
      <td>10160.0</td>
      <td>1050.0</td>
      <td>53.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>claim</td>
      <td>4660861094</td>
      <td>45</td>
      <td>someone shared with me that it would take a 50...</td>
      <td>verified</td>
      <td>active</td>
      <td>931587.0</td>
      <td>171051.0</td>
      <td>67739.0</td>
      <td>4104.0</td>
      <td>2540.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get summary info
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
# Get summary statistics
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


```python
# Values and counts for claim status
data['claim_status'].value_counts()
```

    claim      9608
    opinion    9476
    Name: claim_status, dtype: int64

```python
# Average view count of videos with "claim" status
claims = data[data['claim_status'] == 'claim']

summary_claim = pd.DataFrame({
    'mean': claims.mean(numeric_only=True),
    'median': claims.median(numeric_only=True)
}).applymap(lambda x: f"{x:.2f}")

print(summary_claim)
print('\n')
print('Mean view count claims:', claims['video_view_count'].mean())
print('Median view count claims:', claims['video_view_count'].median())
```

                                   mean         median
    #                           4804.50        4804.50
    video_id              5627264339.77  5607671514.00
    video_duration_sec            32.49          32.00
    video_view_count          501029.45      501555.00
    video_like_count          166373.33      123649.00
    video_share_count          33026.42       17997.50
    video_download_count        2070.95        1139.50
    video_comment_count          691.16         286.00
    
    
    Mean view count claims: 501029.4527477102
    Median view count claims: 501555.0


```python
# Average view count of videos with "opinion" status
opinions = data[data['claim_status'] == 'opinion']

summary_opinion = pd.DataFrame({
    'mean': opinions.mean(numeric_only=True),
    'median': opinions.median(numeric_only=True)
}).applymap(lambda x: f"{x:.2f}")

print(summary_opinion)
print('\n')
print('Mean view count opinions:', opinions['video_view_count'].mean())
print('Median view count opinions:', opinions['video_view_count'].median())
```

                                   mean         median
    #                          14346.50       14346.50
    video_id              5622381723.96  5611857036.00
    video_duration_sec            32.36          32.00
    video_view_count            4956.43        4953.00
    video_like_count            1092.73         823.00
    video_share_count            217.15         121.00
    video_download_count          13.68           7.00
    video_comment_count            2.70           1.00
    
    
    Mean view count opinions: 4956.43224989447
    Median view count opinions: 4953.0

```python
# Counts for each group combination of claim status and author ban status
data.groupby(['claim_status', 'author_ban_status']).count()[['#']]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>#</th>
    </tr>
    <tr>
      <th>claim_status</th>
      <th>author_ban_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">claim</th>
      <th>active</th>
      <td>6566</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>1439</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>1603</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">opinion</th>
      <th>active</th>
      <td>8817</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>196</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>463</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Count, mean and median video share count of each author ban status
data.groupby(['author_ban_status']).agg(
    {'video_view_count': ['count', 'mean', 'median'],
     'video_like_count': ['count', 'mean', 'median'],
     'video_share_count': ['count', 'mean', 'median']
     })
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">video_view_count</th>
      <th colspan="3" halign="left">video_like_count</th>
      <th colspan="3" halign="left">video_share_count</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>author_ban_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>active</th>
      <td>15383</td>
      <td>215927.039524</td>
      <td>8616.0</td>
      <td>15383</td>
      <td>71036.533836</td>
      <td>2222.0</td>
      <td>15383</td>
      <td>14111.466164</td>
      <td>437.0</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>1635</td>
      <td>445845.439144</td>
      <td>448201.0</td>
      <td>1635</td>
      <td>153017.236697</td>
      <td>105573.0</td>
      <td>1635</td>
      <td>29998.942508</td>
      <td>14468.0</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>2066</td>
      <td>392204.836399</td>
      <td>365245.5</td>
      <td>2066</td>
      <td>128718.050339</td>
      <td>71204.5</td>
      <td>2066</td>
      <td>25774.696999</td>
      <td>9444.0</td>
    </tr>
  </tbody>
</table>
</div>


```python
# Create a likes_per_view column
data['likes_per_view'] = data['video_like_count'] / data['video_view_count']

# Create a comments_per_view column
data['comments_per_view'] = data['video_comment_count'] / data['video_view_count']

# Create a shares_per_view column
data['shares_per_view'] = data['video_share_count'] / data['video_view_count']

# Count, mean, and median of each combination of categories of claim status and author ban status
data.groupby(['claim_status', 'author_ban_status']).agg(
    {'likes_per_view': ['count', 'mean', 'median'],
     'comments_per_view': ['count', 'mean', 'median'],
     'shares_per_view': ['count', 'mean', 'median']})
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">likes_per_view</th>
      <th colspan="3" halign="left">comments_per_view</th>
      <th colspan="3" halign="left">shares_per_view</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>claim_status</th>
      <th>author_ban_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">claim</th>
      <th>active</th>
      <td>6566</td>
      <td>0.329542</td>
      <td>0.326538</td>
      <td>6566</td>
      <td>0.001393</td>
      <td>0.000776</td>
      <td>6566</td>
      <td>0.065456</td>
      <td>0.049279</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>1439</td>
      <td>0.345071</td>
      <td>0.358909</td>
      <td>1439</td>
      <td>0.001377</td>
      <td>0.000746</td>
      <td>1439</td>
      <td>0.067893</td>
      <td>0.051606</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>1603</td>
      <td>0.327997</td>
      <td>0.320867</td>
      <td>1603</td>
      <td>0.001367</td>
      <td>0.000789</td>
      <td>1603</td>
      <td>0.065733</td>
      <td>0.049967</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">opinion</th>
      <th>active</th>
      <td>8817</td>
      <td>0.219744</td>
      <td>0.218330</td>
      <td>8817</td>
      <td>0.000517</td>
      <td>0.000252</td>
      <td>8817</td>
      <td>0.043729</td>
      <td>0.032405</td>
    </tr>
    <tr>
      <th>banned</th>
      <td>196</td>
      <td>0.206868</td>
      <td>0.198483</td>
      <td>196</td>
      <td>0.000434</td>
      <td>0.000193</td>
      <td>196</td>
      <td>0.040531</td>
      <td>0.030728</td>
    </tr>
    <tr>
      <th>under review</th>
      <td>463</td>
      <td>0.226394</td>
      <td>0.228051</td>
      <td>463</td>
      <td>0.000536</td>
      <td>0.000293</td>
      <td>463</td>
      <td>0.044472</td>
      <td>0.035027</td>
    </tr>
  </tbody>
</table>
</div>
