import json

data = [
    {
        "intent": "greet",
        "patterns": ["Hello", "Hi", "Hey", "Good morning", "Good evening"],
        "responses": ["Hello! How can I assist you today?", "Hi there! How can I help you?", "Hey! How are you feeling today?"]
    },
    {
        "intent": "ask_definition_substance_abuse",
        "patterns": ["What is substance abuse?", "Define substance abuse", "Can you explain substance abuse?"],
        "responses": ["Substance abuse refers to the harmful or hazardous use of psychoactive substances, including alcohol and illicit drugs."]
    },
    {
        "intent": "ask_symptoms",
        "patterns": ["What are the symptoms of substance abuse?", "How can I tell if someone is abusing drugs?", "Signs of substance abuse"],
        "responses": ["Common symptoms of substance abuse include changes in behavior, physical health issues, neglecting responsibilities, and developing tolerance or withdrawal symptoms."]
    },
    {
        "intent": "ask_treatment_options",
        "patterns": ["What are the treatment options for substance abuse?", "How is substance abuse treated?", "Substance abuse treatment methods"],
        "responses": ["Treatment options for substance abuse include individual therapy, group therapy, family therapy, medication-assisted treatment, and support groups like Alcoholics Anonymous."]
    },
    {
        "intent": "express_feeling_sad",
        "patterns": ["I am feeling sad", "I am depressed", "I feel down"],
        "responses": ["I'm sorry to hear that you're feeling this way. Can you tell me more about what's going on?", "It sounds like you're having a tough time. I'm here to listen.", "It's okay to feel sad sometimes. Do you want to talk about it?"]
    },
    {
        "intent": "provide_support",
        "patterns": ["I need help", "I want to talk to someone", "Can you help me?"],
        "responses": ["Of course, I'm here to help. What do you need support with?", "I'm here for you. How can I assist you today?", "Tell me what's on your mind, and I'll do my best to support you."]
    }
]

with open('intents_add.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)