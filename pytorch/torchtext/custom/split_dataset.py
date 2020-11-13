import random
from torchtext.data import Dataset

# Dataset: Init signature: Dataset(examples, fields, filter_pred=None)
# 如果要分割原来的一个数据集，只需要取出其中的examples和fileds，分割examples之后，用Dataset类创建新的数据集。

# 例如通过TranslationDataset得到了train_dataset，现在想把它分割成训练集和验证集。
all_examples = train_dataset.examples
fields = train_dataset.fields
random.shuffle(all_examples)
index = int(0.9*len(all_examples))

train_examples = all_examples[:index]
val_examples = all_examples[index:]

train_dataset = Dataset(train_examples, fields)
val_dataset = Dataset(val_examples, fields)

