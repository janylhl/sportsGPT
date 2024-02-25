# FitBot: Your AI-Powered Muscle & Lift Companion 🏋️‍♂️

## Personnal purpose of this project :bulb:
As a French data scientist, developing FitBot is a personal journey aimed at deepening my expertise in Natural Language Processing (NLP) and machine learning with LLM's, leveraging the power of Hugging Face Transformers and Keras. This project is more than just building a tool for fitness enthusiasts; it's about challenging myself, pushing the boundaries of my technical skills, and transforming theoretical knowledge into real-world applications. Through FitBot, I seek to blend my passion for fitness with my love for technology, creating a platform that not only enhances users' fitness journeys but also marks a significant milestone in my own professional development and mastery of AI technologies.

## Presentation and ambitions :trophy:
Welcome to FitBot, chatbot tailored for fitness enthusiasts, focusing on the nitty-gritty of musculation and weightlifting. Developed with the cutting-edge Hugging Face Transformers and Keras, FitBot is here to revolutionize your training journey, providing personalized advice, tips, and insights directly at your fingertips.

Why FitBot?

Expert Knowledge, Instantly: Get real-time answers to your muscle-building and weightlifting queries, from workout strategies to nutrition tips.
AI-Driven Support: Powered by advanced NLP and machine learning, FitBot understands your fitness goals and guides you on the right path.
Accessible Anywhere: Whether you're at the gym or home, FitBot is here to support your fitness journey, anytime, anywhere.

Features to develop:
- Personalized Training Recommendations
- Nutritional Guidance for Muscle Gain
- Equipment Suggestions & Technique Tips

Join the Fitness Revolution: Embrace the power of AI and take your training to the next level with FitBot. Whether you're starting out or looking to optimize your performance, FitBot is your go-to source for all things musculation and weightlifting.

Let's Lift Together! 🚀

## Installation :computer:
Follow these steps to set up and run FitBot on your local machine.

### Prerequisites

Ensure you have `git` and `python` (version 3.7 or later) installed on your system.

### Installation Steps

1. **Clone the Repository**
Open a terminal and clone the repository with:
```
git clone [https://github.com/janylhl/sportsGPT.git](https://github.com/janylhl/sportsGPT.git)
cd sportGPT
```
2. **Create a Virtual Environment** (optional but recommended)


3. **Install Dependencies**
Install all required dependencies by running:
```
pip install -r requirements.txt
```

### Launching the Streamlit App
Once installation is complete, you can launch the Streamlit app by executing:
```
streamlit run src/Home.py
```

## Scientific and Technical Journey of the project :book:
My approach to project construction is based on a gradual increase in the complexity of the solution and the tasks performed. First of all, a simple and classic data science project framework was set up for the repo. As far as the GUI is concerned, I decided to make a small streamlit application, since developing a sophisticated interface is absolutely not the primary objective of this project. A first text generation LLM (DistilGPT2) was first used to run the bot. Then I looked at the fine-tuning of such an LLM, and to make it interesting, I chose to train it on a French text generation dataset in order to make it change language. Once this introduction was done, I had to go further on the conversational capabilities of the LLM used.

So the second step was to focus on the conversational aspect. To do this, I had two main options: I could use a model specialized in conversation directly, and work on making it more specialized, or I could stick with a text generation model and build the right environment for its proper use in a chatbot. I'm currently exploring these two approaches with dialoGPT for the conversational network and GPT2 for the text generation LLM, which gives me greater freedom.
I will probably focus my time finetuning a DialoGPT on a french corpus first, by building a specific tokenizer if it's needed and then performe the training. This part have been implemented but i am facing hardware and memory difficulties for now in order to run the full transfer learing experiment, training is about to be perform on a T4 soon. I am also aware that RAG technics could be interesting in this use case to avoid costly fine-tuning.

## Last Updates :tada:
- [2024.02.18] DialoGPT french custom tokenizer and finetuning code
- [2024.02.18] DialoGPT is now used fot the chatbot
- [2024.02.17] Fine Tuning code for DistilGPT2 on french open dataset
- [2024.02.17] Chatbot IHM with StreamLit v.1
- [2024.02.16] Setup the project & text-generation llm running (Distilgpt2)

## Next Updates :wrench:
- DialoGPT finetining on an open french dataset [Currently working on that]
- Build a custom dataset on sports content
- Fine-tuning on a custom dataset about sports content
- Git CI/CD and Hugging Face Hub Delivery
- Replace dialoGPT by a common text-generation LLM in order to have more flexibility