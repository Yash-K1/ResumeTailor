import streamlit as st
import re
import fitz  # PyMuPDF for PDF reading
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import Tool, initialize_agent, AgentType

# ---- PDF text extractor ----
def extract_text_from_pdf(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---- Core LLM setup ----
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

def rewrite_resume_for_jd(resume_text: str, jd_text: str, instructions: str = None) -> str:
    system_prompt = (
        "You are a precise resume writer. Given a candidate resume and a job description, "
        "return a rewritten resume that:\n"
        "1) preserves factual content but rewrites wording to match JD language and keywords.\n"
        "2) emphasizes relevant skills and achievements in bullet form.\n"
        "3) keeps sections: Name (if present), Summary, Skills, Experience, Education.\n"
        "4) make minimal assumptions; if something is unclear, leave it as-is.\n"
        "Return only the resume text (no explanation)."
    )
    if instructions:
        system_prompt += "\nExtra instructions: " + instructions

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Resume:\n{resume_text}\n\nJob Description:\n{jd_text}\n\nProduce rewritten resume:")
    ]
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))

# ---- Skills extractor ----
def extract_skills_from_text(text: str) -> str:
    tokens = re.split(r"[,\n·•\-;]", text)
    candidates = {t.strip() for t in tokens if 2 < len(t.strip()) < 80}
    return ", ".join(sorted(candidates))

# ---- Agent setup ----
rewrite_tool = Tool(
    name="rewrite_resume",
    func=lambda input_str: rewrite_resume_for_jd(*input_str.split("|||", 1)),
    description="Input format: 'RESUME_TEXT ||| JD_TEXT'. Returns a rewritten, JD-tailored resume."
)

skills_tool = Tool(
    name="extract_skills",
    func=lambda text: extract_skills_from_text(text),
    description="Extract skill keywords from text."
)

agent = initialize_agent(
    tools=[rewrite_tool, skills_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

def agent_rewrite(resume_text: str, jd_text: str) -> str:
    return agent.run(
        "Rewrite resume for JD using tools. Return only the final resume text.\n"
        f"Input: {resume_text}|||{jd_text}"
    )

# ---- Streamlit UI ----
st.set_page_config(page_title="Resume Rewriter — Workshop", layout="centered")
st.title("Resume Rewriter — Tailor your resume to a Job Description")

# Initialize session state
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "rewritten_resume" not in st.session_state:
    st.session_state.rewritten_resume = None

uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
jd_text = st.text_area("Job description", height=300)
extra_instr = st.text_input("Extra instructions (optional)")

# Extract PDF text and save in session state
if uploaded_resume:
    st.session_state.resume_text = extract_text_from_pdf(uploaded_resume)

# Display extracted resume if available
if st.session_state.resume_text:
    st.text_area("Extracted Resume", st.session_state.resume_text, height=200)

if st.button("Rewrite Resume"):
    if not st.session_state.resume_text or not jd_text.strip():
        st.error("Please upload a PDF resume and enter a JD.")
    else:
        with st.spinner("Running agent..."):
            jd_skills = extract_skills_from_text(jd_text).split(", ")

            # Display extracted JD skills in a collapsible dropdown
            if jd_skills:
                with st.expander("Detected JD skills / keywords", expanded=False):
                    for skill in jd_skills:
                        if skill.strip():
                            st.markdown(f"- {skill.strip()}")

            try:
                st.session_state.rewritten_resume = agent_rewrite(
                    st.session_state.resume_text, jd_text
                )
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.session_state.rewritten_resume = None

# Display rewritten resume and download button if available
if st.session_state.rewritten_resume:
    st.header("Rewritten Resume")
    st.text_area("Result", value=st.session_state.rewritten_resume, height=500)
    st.download_button("Download as .txt", st.session_state.rewritten_resume, file_name="rewritten_resume.txt")
