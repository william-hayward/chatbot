import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utilities import bag_of_words, tokenize, stem


# model for the neural network used for the chat bot
class Net(nn.Module):
    def __init__(self, input, hidden, classes):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        return output


class Dataset(Dataset):
    def __init__(self):
        self.num_of_samples = len(xtrain)
        self.xdata = xtrain
        self.ydata = ytrain

    def __getitem__(self, idx):
        return self.xdata[idx], self.ydata[idx]

    def __len__(self):
        return self.num_of_samples


with open('responses.json', 'r') as file:
    responses = json.load(file)

words = []
tags = []
xy = []
xtrain = []
ytrain = []
ignore_list = ["?", "!", ".", ","]  # will ignore these characters in the users message

count = 0
# looks within the json file to see how many items are within the intents array and adds the value to the count variable
for i in responses['intents']:
    count = count + 1

# count variable is then used to define the batch_size and output variables. these values therefore don't need to be
# manually changed when more is added to the intents array.
batch_size = count
output = count

rate = 0.001

for i in responses['intents']:
    # adds each tag within the json file to a tags list
    tag = i['tag']
    tags.append(tag)
    for j in i['patterns']:
        # adds each tokenized word within the pattern section of the json file to a words list.
        word = tokenize(j)
        words.extend(word)
        # adds both to the xy list
        xy.append((word, tag))

# stems the words in the list but ignores the characters within ignore_list
words = [stem(i) for i in words if i not in ignore_list]
words = sorted(set(words))
tags = sorted(set(tags))

# generating the data used to train the bot
for (i, j) in xy:
    bag = bag_of_words(i, words)
    tag_label = tags.index(j)
    xtrain.append(bag)
    ytrain.append(tag_label)

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)


input = len(words)
hidden = len(tags)

dataset = Dataset()
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# decides if the users gpu or cpu is used for training the bot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net(input, hidden, output).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=rate)

# lets the user know that training has started
print("Training has started.")

# used for training the bot
for i in range(1000):
    for (j, k) in loader:
        word = j.to(device)
        labels = k.to(device, dtype=torch.int64)
        outputs = model(word)
        loss = criterion(outputs, labels)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


# creates a data dictionary to store the training data
training_data = {
    "state": model.state_dict(),
    "input_size": input,
    "output_size": output,
    "hidden_size": hidden,
    "words": words,
    "tags": tags
}

# saves the contents of the training to a file
torch.save(training_data, "data.pth")


# lets the user know that training is completed.
print("Training Complete.")
