# main.py
#   c2sim main
# by: Noah Syrkis

# imports
import ollama

# globals
model = "mistral:7b-instruct-q2_K"


def prompt_fn(messages):
    response = ollama.chat(model=model, messages=messages)
    messages += [response["message"]]
    return messages


# main
def main(messages=[]):
    for i in range(10):
        print(input_message := input("You: "))  # get input from the human
        messages = prompt_fn(messages + [{"role": "user", "content": input_message}])
        print("Bot: ", messages[-1]["content"], "\n")  # prints bot output
    return messages


if __name__ == "__main__":
    main()
