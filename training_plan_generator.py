import streamlit as st
from datetime import datetime
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# Page config
st.set_page_config(
    page_title="Running Training Plan Generator",
    page_icon="🏃",
    layout="wide"
)

class RunningPlanGenerator:
    """Generate running training plans using GGUF model"""
    
    def __init__(self):
        self.repo_id = "Ashilion/gemma-gguf-q4"
        self.filename = "model.gguf"
        self.model = None
    
    @st.cache_resource
    def load_model(_self):
        """Load GGUF model - cached to avoid reloading"""
        with st.spinner("Loading AI model (first time only)..."):
            # Download model from HuggingFace Hub
            model_path = hf_hub_download(
                repo_id=_self.repo_id,
                filename=_self.filename,
                token=os.getenv("HF_TOKEN")  # Optional: only needed for private repos
            )
            
            # Load model with llama-cpp-python
            model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Context window
                n_threads=4,  # Number of CPU threads
                n_gpu_layers=0,  # 0 for CPU only, increase if you have GPU
                verbose=False
            )
        return model
    
    def generate_plan(self, criteria):
        """Generate training plan based on criteria"""
        if self.model is None:
            self.model = self.load_model()
        
        prompt = self._create_prompt(criteria)
        
#         # Format prompt for chat
#         formatted_prompt = f"""<|system|>
# You are an expert running coach who creates detailed, personalized training plans.</s>
# <|user|>
# {prompt}</s>
# <|assistant|>
# """
        formatted_prompt = (
        "<start_of_turn>user\n"
        "You are an expert running coach who creates detailed, personalized training plans.\n"
        f"{prompt}\n"
        "<end_of_turn>"
        "<start_of_turn>model\n"
    )
        
        # Generate response
        response = self.model(
            formatted_prompt,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "<|user|>"],
            echo=False
        )
        
        plan = response['choices'][0]['text'].strip()
        return plan
    
    def _create_prompt(self, criteria):
        """Create detailed prompt for the model"""
        prompt = f"""Create a detailed {criteria['goal']} training plan with the following specifications:

- Runner Level: {criteria['current_level']}
- Training Duration: {criteria['timeframe']} weeks
- Days per week: {criteria['days_per_week']}"""
        
        if criteria.get('target_time'):
            prompt += f"\n- Target Finish Time: {criteria['target_time']}"
        
        if criteria.get('current_distance'):
            prompt += f"\n- Current Long Run Distance: {criteria['current_distance']}"
        
        prompt += """

Please structure the plan with:
1. Weekly breakdown with specific workouts
2. Different types of runs (easy, tempo, intervals, long runs)
3. Rest days
4. Progressive mileage increase
5. Taper period before race day

Format each week separately with the types of run and distance.
Then provide more details for each interval and tempo run, including pace targets based on the goal time.
"""
        
        return prompt


# Streamlit App
def main():
    st.title("🏃 AI Running Training Plan Generator")
    st.markdown("Generate personalized running training plans powered by AI")
    
    # Sidebar for inputs
    st.sidebar.header("Training Plan Criteria")
    
    athlete_name = st.sidebar.text_input("Your Name", "Runner")
    
    goal = st.sidebar.selectbox(
        "Goal Race",
        ["5K", "10K", "Half Marathon", "Marathon"]
    )
    
    current_level = st.sidebar.selectbox(
        "Current Level",
        ["Beginner", "Intermediate", "Advanced"]
    )
    
    timeframe = st.sidebar.slider(
        "Training Duration (weeks)",
        min_value=4,
        max_value=24,
        value=12,
        step=1
    )
    
    days_per_week = st.sidebar.slider(
        "Days per week",
        min_value=3,
        max_value=7,
        value=4,
        step=1
    )
    
    # Optional fields
    with st.sidebar.expander("Optional Details"):
        target_time = st.text_input(
            "Target Time (e.g., 45:00, 1:30:00)",
            ""
        )
        
        current_distance = st.text_input(
            "Current Long Run Distance (e.g., 5 miles, 10K)",
            ""
        )
    
    criteria = {
        "goal": goal,
        "current_level": current_level,
        "timeframe": timeframe,
        "days_per_week": days_per_week,
        "target_time": target_time if target_time else None,
        "current_distance": current_distance if current_distance else None
    }
    
    # Generate button
    if st.sidebar.button("Generate Training Plan", type="primary", use_container_width=True):
        # Initialize generator
        generator = RunningPlanGenerator()
        
        # Generate plan
        with st.spinner("Generating your personalized training plan..."):
            try:
                plan = generator.generate_plan(criteria)
                
                # Store in session state
                st.session_state.plan = plan
                st.session_state.criteria = criteria
                st.session_state.athlete_name = athlete_name
                st.success("✅ Training plan generated successfully!")
            except Exception as e:
                st.error(f"Error generating plan: {str(e)}")
    
    # Display results
    if 'plan' in st.session_state:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Your Training Plan")
            st.markdown(st.session_state.plan)
        
        with col2:
            st.subheader("Plan Summary")
            st.info(f"""
            **Goal:** {st.session_state.criteria['goal']}
            
            **Level:** {st.session_state.criteria['current_level']}
            
            **Duration:** {st.session_state.criteria['timeframe']} weeks
            
            **Training Days:** {st.session_state.criteria['days_per_week']} per week
            """)
            
            # Download as text
            st.download_button(
                label="📄 Download as Text",
                data=st.session_state.plan,
                file_name=f"training_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        # Welcome message
        st.info("👈 Fill in your training criteria in the sidebar and click 'Generate Training Plan' to begin!")
        
        st.markdown("""
        ### How it works:
        1. **Enter your details** in the sidebar (name, goal, level, etc.)
        2. **Click Generate** to create your personalized plan
        3. **Review** your custom training schedule
        4. **Download** as text 
        
        ### Requirements:
        ```bash
        pip install streamlit llama-cpp-python huggingface-hub
        ```
        """)


if __name__ == "__main__":
    main()