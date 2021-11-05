# CODES

## Download Dataset

```python
# download kaggle.js API token from kaagle.com> accounts> API token and upload here

from google.colab import files, output
files.upload()

!pip install -q kaggle

!mkdir -p ~/.kaggle/
!cp kaggle.json ~/.kaggle/

#change permission of the file

!chmod 600 ~/.kaggle/kaggle.json

!mkdir dir_name

%cd dir_name

!kaggle datasets download praveengovi/coronahack-chest-xraydataset(dataset_link)


!kaggle datasets download mlg-ulb/creditcardfraud

%pwd

!unzip coronahack-chest-xraydataset.zip
output.clear()

#remove zipped dataset

!rm coronahack-chest-xraydataset.zip
```



