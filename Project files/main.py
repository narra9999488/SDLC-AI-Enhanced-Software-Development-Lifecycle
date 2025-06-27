from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import os, json, torch, bcrypt

# --- Environment Setup ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Model Setup ---
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct", use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-3.3-2b-instruct",
    torch_dtype=torch.float32,
    use_auth_token=True
)

# --- User Data Functions ---
USER_FILE = "users.json"
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# --- In-memory Chat ---
chat_history = []
prompts = [
#     ("Generate Software Requirements Specification", "Create a basic SRS template"),
#     ("Explain Agile Model", "Give an overview of the Agile process"),
#     ("Difference between Waterfall and Agile", "Compare two popular SDLC models"),
#     ("List SDLC Phases", "Enumerate the phases in SDLC")
       ("Generate SRS", "Create a sample Software Requirement Specification document."),
        ("Explain Agile Model", "Give an overview of the Agile SDLC model."),
        ("Waterfall vs Agile", "Compare Waterfall and Agile development approaches."),
        ("List SDLC Phases", "What are the phases in Software Development Lifecycle?"),
        ("Create Test Plan", "Generate a test plan for a software project."),
        ("Write Use Case", "Example of a use case for a login system."),
        ("Explain Spiral Model", "Describe the Spiral Model in SDLC."),
        ("What is DevOps?", "Give a brief explanation of DevOps."),
        ("Explain MVP", "What is a Minimum Viable Product in software?")

 ]

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_history": chat_history,
        "prompts": prompts
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    users = load_users()
    if username in users and verify_password(password, users[username]):
        request.session['user'] = username
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Invalid credentials"
    })

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")

@app.post("/send", response_class=HTMLResponse)
async def send_message(request: Request, user_input: str = Form(...)):
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output.replace(user_input, "").strip()

    chat_history.append(("user", user_input))
    chat_history.append(("bot", response))

    return RedirectResponse("/", status_code=303)

@app.get("/clear", response_class=HTMLResponse)
async def clear_chat(request: Request):
    chat_history.clear()
    return RedirectResponse("/", status_code=303)

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    user_input = f"Review this document:\n{text[:1500]}"
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output.replace(user_input, "").strip()

    chat_history.append(("user", "Uploaded Document"))
    chat_history.append(("bot", response))

    return RedirectResponse("/", status_code=303)

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
def signup(request: Request, username: str = Form(...), password: str = Form(...)):
    users = load_users()
    if username in users:
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "error": "Username already exists."
        })
    users[username] = hash_password(password)
    save_users(users)
    request.session['user'] = username
    return RedirectResponse("/", status_code=303)
