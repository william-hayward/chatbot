import random
import json
import torch
from bot_training import Net
from utilities import bag_of_words, tokenize


def run():
    with open("responses.json", "r") as file:
        responses = json.load(file)

    # decides if the users gpu or cpu is used for training the bot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # extracting data from the pth file
    data = torch.load("data.pth")

    input_size = data["input_size"]
    hidden = data["hidden_size"]
    output = data["output_size"]
    words = data["words"]
    tags = data["tags"]
    state = data["state"]

    # defining the neural network model
    model = Net(input_size, hidden, output).to(device)
    model.load_state_dict(state)
    model.eval().to(device)

    # prints this message at the start of the chat as a welcome message to the user.
    print("-----------------\nHello! This is customer support. How can i help you? (type quit to exit)\n")

    while True:
        user_sentence = input("You: ")
        # if the user types quit. the program will end
        if user_sentence.lower() == "quit":
            break

        # tokenizes the users input and calls the bag_of_words function
        user_sentence = tokenize(user_sentence)
        x = bag_of_words(user_sentence, words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(device)

        # generates the probabilities using the users sentence
        bot_output = model(x).to(device)
        y, predicted = torch.max(bot_output, dim=1)

        # generates the index of the predicted tag related to the users input
        tag = tags[predicted.item()]

        # using the index generated this selects the corresponding probability and saves it to a probability variable
        probabilities = torch.softmax(bot_output, dim=1)
        probability = probabilities[0][predicted.item()]

        # if the returned probability is 0.75 or higher, the program will respond with a random choice of output from
        # the chosen tag. this allows the bot to give the best response to a user input that is not hard coded in the
        # json
        if probability.item() >= 0.75:
            for i in responses["intents"]:
                if tag == i["tag"]:
                    output_message = random.choice(i["responses"])
                    print(f'Bot: {output_message}\n')

        # if the probability is less than 0.75 the bot will respond with a message stating it doesn't understand
        else:
            print("Bot: I don't understand. Could you please make your query more clear? Thank you.\n")


if __name__ == "__main__":
    run()
