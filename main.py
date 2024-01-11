import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download

class StreamHandler(BaseCallbackHandler):

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    @st.cache_resource
    def create_chain(system_prompt):
        (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                  "mistral-7b-instruct-v0.1.Q4_0.gguf")
        model_path = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=512,
            top_p=1,
            # callback_manager=callback_manager,
            n_gpu_layers=1,
            # n_batch=512,
            # n_ctx=4096,
            stop=["[INST]"],
            verbose=False,
            streaming=True
            )
        
        context = """
        <s>[INST]{}[/INST]</s>
            Elliot Willner, 20, curious, CS at Georgia Tech
            Interest: tennis, games, computers, new food
            From Atlanta. studied abroad, France
            Skills: ML, LLMs, python
        [INST]{}[/INST]
        """.format(system_prompt, "{question}")

        #He has lived in Atlanta his whole life. When he was young, he was interested in tennis, video games, and computers.

        # Create prompt
        prompt = PromptTemplate(template=context, input_variables=["question"])

        llm_chain = prompt | llm

        return llm_chain
    
    st.set_page_config(
    page_title="My personal website AI"
    )

    st.header("My personal website AI")

    system_prompt = st.text_area(
        label="System Prompt",
        value= "Act as a human Elliot Willner answering questions about yourself.",
        key="system_prompt")
    
    llm_chain = create_chain(system_prompt)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask questions about me here!"}
            ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_prompt := st.chat_input("Your message here", key="user_input"):

        # Add input to the session state
        st.session_state.messages.append(
            {"role": "user", "content": user_prompt}
            )

        # Add input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        response = llm_chain.invoke({"question": user_prompt})

        # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
            )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)