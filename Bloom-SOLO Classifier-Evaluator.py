from langchain.chains import load_chain
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain, ConversationChain, ConversationalRetrievalChain, PALChain, LLMMathChain, LLMBashChain, LLMCheckerChain, LLMRequestsChain, LLMSummarizationCheckerChain, create_tagging_chain
from langchain.chains.router import MultiPromptChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
import streamlit as st

local_path = (
    "D:/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin"  
)
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(model=local_path, callbacks=callbacks, backend="gptj", verbose=True)


SOLO_schema = {
    "properties": {
        "SOLO_Level": {"type": "string", "enum": ["Pre-structural", "Unistructural", "Multistructural", "Relational", "Extended Abstract"]},
        "SOLO_Numerical_Taxonomy": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "describes how good the answer is based on the solo taxonomy where Pre-structural = 1, Unistructural= 2, Multistructural = 3, Relational= 4, Extended Abstract = 5",
        },
    },
    "required": ["SOLO_Level", "SOLO_Numerical_Taxonomy"],
}

SOLOEvaluator_Chain = create_tagging_chain(SOLO_schema, llm)

BloomEvaluator_schema = {
    "properties": {
        "Bloom_Objective": {"type": "string", "enum": ["Knowledge", "Comprehension", "Application", "Analysis", "Evaluation", "Synthesis"]},
        "BLoom_Numerical_Taxonomy": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "describes the learning depth that the student reached based on their given answers where Knowledge = 1, Comprehension = 2, Application = 3, Analysis = 4, Evaluation = 5, Synthesis = 6",
        },
    },
    "required": ["Bloom_Objective", "BLoom_Numerical_Taxonomy"],
}

BloomEvaluator_Chain = create_tagging_chain(BloomEvaluator_schema, llm)

Bloom_Classifier = {
    "properties": {
        "Bloom_Objective": {"type": "string", 
                            "enum": ["Knowledge", "Comprehension", "Application", "Analysis", "Evaluation", "Synthesis"],
                            "description": "classifies a statement into the corresponding Bloom Taxonomy depending on the compatibility of concepts discussed in the statement with each Bloom learning objective (selective comparison)"
                            },
    },
    "required": ["Bloom_Objective"],
}


BloomClassifer_Chain = create_tagging_chain(Bloom_Classifier, llm)

 

interface_choice = st.radio("Choose a corresponding functionality", ["Classification", "Evaluation"])

if interface_choice == "Evaluation":
    taxonomy_choice = st.radio("Choose Learning Taxonomy Mode", ["Bloom Taxonomy", "SOLO Taxonomy"])
    if taxonomy_choice == "Bloom Taxonomy":
        answer = st.text_area("Insert your answer here!")
        resp = BloomEvaluator_Chain.run(answer)
        st.write(resp)
    if taxonomy_choice == "SOLO Taxonomy":
        answer = st.text_area("Insert your answer here!")
        resp = SOLOEvaluator_Chain.run(answer)
        st.write(resp)
        
if interface_choice == "Classification":
    statement = st.text_area("Insert your statement here!")
    resp = BloomClassifer_Chain.run(statement)
    st.write(resp)







