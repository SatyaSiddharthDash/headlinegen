import pandas as pd
from sklearn.model_selection import train_test_split


random_state = 100

data = pd.read_csv("~/headlinegen/data/nytime_front_page.csv")
data['title'] = data['title'].apply(lambda x: ' '.join(x.split(' ')[:-5]))

lens = data["content"].apply(lambda x: len(x.split(" "))).nlargest(10)

print(
    f'max_input_len = {data["content"].apply(lambda x: len(x.split(" "))).min()}')
print(
    f'max_output_len = {data["title"].apply(lambda x: len(x.split(" "))).max()}')

print(lens)

# train, valid_test = train_test_split(data,
#                                      test_size=0.2,
#                                      random_state=random_state,
#                                      shuffle=True)
# valid, test = train_test_split(valid_test,
#                                test_size=0.5,
#                                random_state=random_state,
#                                shuffle=True)

# print(train.shape, valid.shape, test.shape)

# for dataset, prefix in zip([train, valid, test], ['train', 'val', 'test']):
#     for columnname, suffix in zip(['content', 'title'], ['source', 'target']):
#         filename = "/Users/satyasiddharthdash/headlinegen/data/nytimes/" + prefix + '.' + suffix
#         with open(filename, 'w') as outfile:
#             outfile.write(dataset[columnname].str.cat(sep='\n'))
