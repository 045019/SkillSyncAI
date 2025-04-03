# 1.0 SimpleDirectoryReader can also load metadata from a dictionary
#     https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PagedCSVReader

# 1.1 The Settings is a bundle of commonly used resources used 
#     during the indexing and querying stage in a LlamaIndex workflow/application.
from llama_index.core import Settings

# 1.2 If using LocalAI
# https://docs.llamaindex.ai/en/stable/examples/llm/localai/
#from llama_index.llms.openai_like import OpenAILike

# 1.3 Ollama related
# https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# 1.4 Vector store related
import chromadb
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

# Model
from langchain_community.tools.tavily_search import TavilySearchResults
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# 1.6 Misc
import pandas as pd



# 2.0 Define embedding function

embed_model= OllamaEmbedding(
                                    model_name="nomic-embed-text",      # Using foundational model may be overkill
                                    base_url="http://localhost:11434",
                                    #dimensions=512,
                                    #ollama_additional_kwargs={"mirostat": 0},
                                  )
Settings.embed_model = embed_model

# 2.1 Settings can set the global configuration. Local configurations (transformations, LLMs, embedding models).
#     These can be passed directly into the interfaces that make use of them.
from llama_index.llms.mistralai import MistralAI
llm = MistralAI(api_key="txuXbijTrapzLjfKQdEGSXWqSH6Ovdni")
Settings.llm = llm



# 3.0 Reading data in pandas
#     It has nothing to do with subsequent analysis/usage


file_path = ('https://raw.githubusercontent.com/045019/SkillSyncAI/refs/heads/main/trim%20data.csv') # insert the path of the csv file
df = pd.read_csv(file_path)

# 3.1 Preview the csv file
df.head()




# 4.0 PagedCSVReader displays each row in an LLM-friendly format. Each row as a separate document.
csv_reader = PagedCSVReader()

# 4.1
reader = SimpleDirectoryReader( 
                                input_files=[file_path],
                                file_extractor= {".csv": csv_reader}
                               )

# 4.2
docs = reader.load_data()



# 5.0 Create client and a new collection
#     The following is  in-memory database and NOT a persistent collection.
#     chroma_client = chromadb.EphemeralClient()

# 5.1 This creates persistent collection. A folder by name of chromadb
#     is created and below that a chroma.sqlite3 database exists:

chroma_client = chromadb.PersistentClient(path="https://raw.githubusercontent.com/045019/SkillSyncAI/refs/heads/main/chromadb")


# 5.2 Check if collection exists. If so delete it.
#     Collections are the grouping mechanism for embeddings, documents, and metadata.
#     Chromadb can have multiple collections

if 'datastore' in chroma_client.list_collections():
    chroma_client.delete_collection("datastore")
    chroma_collection = chroma_client.create_collection("datastore")  
else:
    # Create collection afresh
    chroma_collection = chroma_client.create_collection("datastore")   

# 5.3 Get collection information:
chroma_collection


# 6.0 Set up a blank ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 6.1
storage_context = StorageContext.from_defaults(vector_store=vector_store)



# 6.2 Takes docs and storage context:
#     Repeating this operation, doubles the number of vectors/records in the collection

index = VectorStoreIndex.from_documents(
                                         docs,
                                         storage_context=storage_context,
                                         show_progress= False                 # Show progress bar
                                        )



query_engine = index.as_query_engine()


def extract_name_roll_job_llm(query: str) -> dict:
    prompt = f"""
You are an intelligent input parser.

Extract the following fields from the user's message:
- Full Name
- Roll Number
- Desired Job Role

Return the result in **valid JSON format** like this:
{{
  "name": "Full Name",
  "roll": "Roll Number",
  "job": "Job Role"
}}

If any field is not mentioned, return its value as null.

USER QUERY:
\"\"\"{query}\"\"\"
"""
    import json
    raw = llm.complete(prompt).text.strip()

    # 🔧 Fix: Strip markdown code block if present
    if raw.startswith("```"):
        raw = raw.strip("`")  # removes all backticks
        if "json" in raw:
            raw = raw.replace("json", "", 1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"name": None, "roll": None, "job": None}


def validate_student_from_profile(profile_response: str, expected_name: str, expected_roll: str) -> str:
    prompt = f"""
Validate the following student profile.

Expected Name: {expected_name}
Expected Roll Number: {expected_roll}

STUDENT PROFILE:
\"\"\"{profile_response}\"\"\"

👉 Only reply:
- ✅ Valid Student (if name and roll number match)
- ❌ Invalid Student (if either name or roll number is missing or mismatched)
"""
    return llm.complete(prompt).text.strip()



def infer_skills_from_profile(profile_text: str) -> list:
    prompt = f"""
You are a job counselor AI.

Given the following student academic profile, infer **10 specific and realistic job-relevant skills** the student likely possesses based on:

- Degree
- Specialization
- Graduation percentage or CGPA
- Career aspiration (if mentioned)

🎯 Format the response as a simple bullet list.  
❌ Do NOT include any explanation, repetition of profile, or generic skills.

STUDENT PROFILE:
{profile_text}
"""
    response = llm.complete(prompt).text.strip()
    return [line.strip("-• ").strip() for line in response.splitlines() if line.strip()]



search = TavilySearchResults(k=5, tavily_api_key="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")
def fetch_real_time_job_skills(job: str) -> list:
    search_results = search.run(f"Top in-demand skills for {job} from LinkedIn, Glassdoor, and Indeed in 2024")

    prompt = f"""
Based on the following search results, list the **10 most in-demand skills** for the job role: {job}

Search Results:
{search_results}

Respond with one skill per line, no explanations.
"""
    response = llm.complete(prompt).text.strip()
    return [line.strip("-• ").strip() for line in response.splitlines() if line.strip()]



def compare_skills(student_skills: list, job_skills: list, job: str) -> str:
    # Make sure all skills are strings
    student_skills = [str(skill) for skill in student_skills]
    job_skills = [str(skill) for skill in job_skills]
    prompt = f"""
Compare the student's skills with the job requirements for the role: {job}

STUDENT SKILLS: {', '.join(student_skills)}
JOB SKILLS: {', '.join(job_skills)}

Give:
- ✅ Matching Skills
- ❌ Missing Skills (with Priority: High/Medium/Low)
- 📘 Suggestions to improve

At the end, list missing skills like:
[MISSING SKILLS]: skill1, skill2
"""
    return llm.complete(prompt).text.strip()

def extract_missing_skills(skill_gap_text: str) -> list:
    prompt = f"""
You are a smart parser.

From the following **Skill Gap Analysis**, extract only the missing skills listed under the [MISSING SKILLS] section.

Skill Gap Report:
\"\"\"{skill_gap_text}\"\"\"

👉 Return only a list like:
- Skill A
- Skill B
- Skill C

Do not include any explanation, priority, or extra formatting.
"""
    response = llm.complete(prompt).text.strip()

    # Clean into a Python list
    return [line.strip("-• ").strip() for line in response.splitlines() if line.strip().startswith("-")]



def recommend_courses(skill: str) -> str:
    # 🔍 Step 1: Use Tavily to fetch real web data
    search_results = search.run(
        f"Top online courses to learn {skill} from Coursera, Udemy, or edX in 2024. Include course titles, platforms, and links."
    )

    # 🤖 Step 2: Let Mistral summarize and format course recommendations
    prompt = f"""
You are a smart course advisor AI.

From the following search results, extract and recommend **3 to 5 real, high-quality courses** for learning **{skill}**.

Instructions:
- Extract actual course titles and platforms (Coursera, Udemy, edX)
- If real links are mentioned, include them
- ⚠️ Do NOT make up course links — only use links you find in the search results
- Format your answer as a Markdown table:

| Course Name | Platform | Link |

Search Results:
\"\"\"
{search_results}
\"\"\"
"""
    return llm.complete(prompt).text.strip()




def recommend_jobs(profile_text: str, job: str) -> str:
    # 🔍 1. Fetch job search results using Tavily
    search_query = (
        f"Latest job openings for {job} in India 2024 "
        f"with eligibility criteria, job descriptions, and company names"
    )
    search_results = search.run(search_query)

    # 🤖 2. Prompt Mistral to check eligibility using multiple parameters
    prompt = f"""
You are an AI career advisor.

Your task is to analyze job listings and check whether a student is **eligible** for each based on their complete academic profile.

👤 STUDENT PROFILE:
{profile_text}

You must check the following eligibility criteria from the student profile:
- 🎓 Degree
- 🧠 Specialization
- 📊 CGPA or Graduation Percentage
- 📆 Graduation Year (if available or inferred)
- 👤 Age or Birthdate
- 🎯 Career Aspiration (if mentioned)

📝 JOB LISTINGS (from live web search):
{search_results}

For each job, do the following:
1. Extract job eligibility requirements from the description.
2. Compare with the student profile.
3. Determine if the student is eligible.
4. Return results in the following format:

- **Job Title** – *Company Name*
  - 🎓 Eligibility: ✅ Eligible / ❌ Not Eligible
  - 📝 Reason: Explain the eligibility decision clearly (e.g., degree mismatch, CGPA too low, age limit exceeded)
  - 🔗 Link (if mentioned in search results)

Only list 5-10 top jobs.

Be precise and base your decision on **all relevant parameters**, not just the degree.
"""
    return llm.complete(prompt).text.strip()

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

extract_profile_tool           = FunctionTool.from_defaults(fn=extract_name_roll_job_llm)
infer_skills_tool              = FunctionTool.from_defaults(fn=infer_skills_from_profile)
fetch_job_skills_tool          = FunctionTool.from_defaults(fn=fetch_real_time_job_skills)
compare_skills_tool            = FunctionTool.from_defaults(fn=compare_skills)
extract_missing_skills_tool    = FunctionTool.from_defaults(fn=extract_missing_skills)
recommend_courses_tool         = FunctionTool.from_defaults(fn=recommend_courses)
recommend_jobs_tool            = FunctionTool.from_defaults(fn=recommend_jobs)


career_tools = [
    extract_profile_tool,
    infer_skills_tool,
    fetch_job_skills_tool,
    compare_skills_tool,
    extract_missing_skills_tool,
    recommend_courses_tool,
    recommend_jobs_tool
]

agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=career_tools,
    llm=llm,
    verbose=False
)

agent = AgentRunner(agent_worker)

cached_profile = {
    "name": None,
    "roll": None,
    "job": None,
    "profile_response": None
}



def run_career_agent(user_query: str):
    global cached_profile

    # 🔍 Step 1: Extract name, roll number, and job from the query using LLM
    result = extract_name_roll_job_llm(user_query)
    name = result.get("name") or cached_profile["name"]
    roll = result.get("roll") or cached_profile["roll"]
    job  = result.get("job") or cached_profile["job"]

    if not name or not roll:
        return "⚠️ Please provide your name and roll number to continue."

    # 🧠 Step 2: Validate if this is a new student (different from cached one)
    is_new_user = (name != cached_profile["name"]) or (roll != cached_profile["roll"])

    if is_new_user:
        profile_query = f"Give me all details of the student with Roll Number {roll} and full name as '{name}'"
        profile_response = query_engine.query(profile_query)

        # ✅ Step 3: Verify that the response contains valid data
        validation = validate_student_from_profile(str(profile_response), name, roll)
        if "❌" in validation:
            return "❌ This user is not found in the system. Please check your name or roll number."

        # 💾 Step 4: Save to cache if valid
        cached_profile.update({
            "name": name,
            "roll": roll,
            "job": job,
            "profile_response": profile_response
        })

    # 💬 Step 5: Run the agent with the enriched query and profile
    profile_text = cached_profile["profile_response"]
    enriched_query = f"""
USER QUERY:
{user_query}

STUDENT PROFILE:
{profile_text}
"""
    return agent.chat(enriched_query).response.strip().replace("\n\n", "\n")


