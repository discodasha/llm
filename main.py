import os, yaml
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.utilities import WikipediaAPIWrapper
from langchain import LLMMathChain
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
import tiktoken
from langchain.agents.agent_toolkits.openapi import planner
from langchain.requests import RequestsWrapper
from langchain.tools.json.tool import JsonSpec
from langchain.agents import create_openapi_agent
from langchain.agents.agent_toolkits import OpenAPIToolkit
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper



#llm = OpenAI(temperature=0.5)
llm = OpenAI(model_name="gpt-4", temperature=0.0)
  
   
def test(text):
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    print(prompt.format(product=text))
    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run(text))


def toolsTest(text):
    wikipedia = WikipediaAPIWrapper()
    tools = [
        Tool(
            name="Wiki",
            func=wikipedia.run,
            description="useful for when you need to answer questions about current events"
        ),
    ]
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools.append(
        Tool (
            name="math",
            func= llm_math_chain.run,
            description="calculatiion"
        )
    )
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    answer = agent.run(
        "How much albums James Blake has? \
            What is that number raised to the 2 power?")
    print(answer)
    return answer


def reporter():
    with open("/home/dasha/Downloads/reporter.yml") as f:
        raw_openai_api_spec = yaml.load(f, Loader=yaml.FullLoader)
    openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
    endpoints = [
        (route, operation)
        for route, operations in raw_openai_api_spec["paths"].items()
            for operation in operations
                if operation in ["get", "post"]
    ]
    # enc = tiktoken.encoding_for_model('text-davinci-003')
    # def count_tokens(s): return len(enc.encode(s))
    # count_tokens(yaml.dump(raw_openai_api_spec))
    
    # requests_wrapper = RequestsWrapper()
    # spotify_agent = planner.create_openapi_agent("app.jaicp.com", openai_api_spec, requests_wrapper, llm)
    # user_query = "how to change a label of message"
    # return spotify_agent.run(user_query)

    json_spec=JsonSpec(dict_=raw_openai_api_spec, max_value_length=4000)
    openai_requests_wrapper=RequestsWrapper()

    openapi_toolkit = OpenAPIToolkit.from_llm(OpenAI(temperature=0), json_spec, openai_requests_wrapper, verbose=True)
    openapi_agent_executor = create_openapi_agent(
        llm=OpenAI(temperature=0),
        toolkit=openapi_toolkit,
        verbose=True
    )   
    return openapi_agent_executor.run("Make a post request to get sessions where 'SESSION_ID' is 1231455 ")


if __name__ == '__main__':
    #test("crochet bags")
    # res = toolsTest("")
    # print(type(res))
    # print(res[-17:])
    print(reporter())
