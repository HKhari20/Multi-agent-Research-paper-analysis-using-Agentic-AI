import os
import google.generativeai as genai

# Recommended models in order of priority/availability
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

AGENT_DEFINITIONS = {
    "summarizer": {
        "id": "summarizer",
        "name": "Summarizer",
        "title": "Lead Academic Summarizer",
        "icon": "📝",
        "color": "#FF6B6B",
        "theme": "coral",
        "tagline": "High-level essence, core problem & key discoveries",
        "role_prompt": (
            "You are an Expert Academic Summarizer and Senior Research Analyst. "
            "Your objective is to provide an articulate, high-level, yet thorough executive distillation of the research paper."
        ),
        "task": (
            "Analyze the paper and provide a structured summary:\n"
            "1. **Core Problem Statement**: What critical challenge or gap in the field is this paper addressing?\n"
            "2. **Proposed Innovation / Solution**: What is the primary novelty, framework, or architecture introduced?\n"
            "3. **Key Findings & Results**: What were the quantifiable outcomes and experimental breakthroughs?\n"
            "4. **Impact & Relevance**: Why does this research matter to the scientific and industrial community?"
        )
    },
    "methodologist": {
        "id": "methodologist",
        "name": "Methodologist",
        "title": "Methodology & Architecture Specialist",
        "icon": "🔬",
        "color": "#00F2FE",
        "theme": "cyan",
        "tagline": "Algorithms, mathematical formulation & experiment design",
        "role_prompt": (
            "You are a Strict Data Scientist, Machine Learning Architect, and Methodological Auditor. "
            "Your objective is to dissect technical mechanisms, mathematical representations, data flows, and experimental rigs."
        ),
        "task": (
            "Extract and document the technical foundations of the paper:\n"
            "1. **Architectures & Algorithmic Design**: Detail the exact algorithms, pipeline steps, and structural designs.\n"
            "2. **Mathematical Formulations**: Note equations, loss functions, optimization objectives, and formal representations.\n"
            "3. **Datasets & Preprocessing**: Which benchmarks, corpora, or experimental cohorts were used?\n"
            "4. **Training & Hyperparameters**: Note compute resources, batch sizes, learning rates, epochs, or hardware where mentioned.\n"
            "5. **Evaluation Metrics**: How was performance quantitatively measured (e.g., BLEU, Accuracy, F1, latency)?"
        )
    },
    "critic": {
        "id": "critic",
        "name": "Critic",
        "title": "Peer Reviewer & Risk Auditor",
        "icon": "⚖️",
        "color": "#FEE140",
        "theme": "amber",
        "tagline": "Limitations, biases, reproducibility risks & edge cases",
        "role_prompt": (
            "You are an Elite Peer Reviewer for top-tier AI and science conferences (NeurIPS/ICLR/Nature). "
            "You are uncompromising, objective, and analytically rigorous. You seek hidden flaws, overstated claims, and reproducibility barriers."
        ),
        "task": (
            "Conduct an adversarial yet constructive peer review:\n"
            "1. **Methodological Weaknesses & Assumptions**: Where does the theoretical foundation or experimental design rely on brittle assumptions?\n"
            "2. **Benchmark Bias & Potential Leakage**: Are the comparisons fair, baselines modern, and datasets representative?\n"
            "3. **Generalizability & Failure Modes**: Under what edge cases or environmental shifts would this method fail?\n"
            "4. **Reproducibility & Transparency**: Is enough detail provided for an independent researcher to replicate the findings?"
        )
    },
    "synthesizer": {
        "id": "synthesizer",
        "name": "Synthesizer",
        "title": "Executive Synthesizer & Editorial Lead",
        "icon": "🧠",
        "color": "#B721FF",
        "theme": "violet",
        "tagline": "Cohesive executive briefing, verdict & strategic roadmaps",
        "role_prompt": (
            "You are the Lead Editor-in-Chief of a premier AI Research Journal and Chief AI Technology Strategist. "
            "You synthesize disparate evaluations into an authoritative, strategic intelligence briefing."
        ),
        "task": (
            "Synthesize the outputs from the Summarizer, Methodologist, and Critic into an authoritative Executive Intelligence Dossier:\n"
            "1. **Executive Overview**: High-level distillation uniting the problem, innovation, and outcome.\n"
            "2. **Technical Feasibility & Rigor**: Balanced synthesis of methodological strength versus critical weaknesses.\n"
            "3. **Peer Review Verdict**: Provide a clear recommendation (e.g., Strongly Accept, Accept with Reservations, Needs Major Revision) with justification.\n"
            "4. **Strategic Takeaways & Industry Recommendations**: How engineering teams and researchers should act on this work."
        )
    }
}


def test_api_key(api_key: str):
    """
    Validates the Gemini API key and retrieves available generation models.
    Returns: (is_valid: bool, models_or_error: list[str] | str)
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty."
    
    clean_key = api_key.strip()
    try:
        genai.configure(api_key=clean_key)
        # Fast lightweight check to list models
        available = []
        for model in genai.list_models():
            if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                # Clean name: remove 'models/' prefix
                m_name = model.name.replace('models/', '')
                available.append(m_name)
        
        # Sort priority models first if found
        prioritized = [m for m in DEFAULT_MODELS if m in available]
        others = [m for m in available if m not in prioritized]
        result_models = prioritized + others
        
        if not result_models:
            result_models = DEFAULT_MODELS

        return True, result_models
    except Exception as e:
        return False, str(e)


class MultiAgentSystem:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """Initializes the multi-agent system with the provided API key."""
        clean_key = (api_key or "").strip()
        genai.configure(api_key=clean_key)
        
        # Clean model name
        clean_model_name = (model_name or "gemini-2.5-flash").replace("models/", "").strip()
        # Fallback if invalid model name was carried over
        if clean_model_name == "gemini-3.5-flash":
            clean_model_name = "gemini-2.5-flash"
            
        self.model_name = clean_model_name
        self.model = genai.GenerativeModel(clean_model_name)

    def run_agent(self, name: str, role_prompt: str, task: str, context: str) -> str:
        """
        Simulates an independent agent processing a specific task based on its role.
        """
        full_prompt = (
            f"You are acting as the following role: {role_prompt}\n\n"
            f"Task: {task}\n\n"
            f"Context Data (Research Paper Excerpt):\n{context}\n\n"
            f"Please provide your analysis based solely on your role and the context provided. "
            f"Format your response with clear, professional markdown headings, bullet points, and bold text."
        )
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            # Try automatic fallback to gemini-1.5-flash if primary model encounters error
            if self.model_name != "gemini-1.5-flash":
                try:
                    fallback_model = genai.GenerativeModel("gemini-1.5-flash")
                    fallback_response = fallback_model.generate_content(full_prompt)
                    return fallback_response.text
                except Exception:
                    pass
            return f"⚠️ **Error in {name} Agent:** {str(e)}"
