# interactive.py - SAM Interactive Interface
import sys
import json
import time
import threading
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SAMInteractive:
    """Interactive interface for SAM"""
    
    def __init__(self, model, config=None, save_conversations=True):
        self.model = model
        self.config = config or {}
        self.save_conversations = save_conversations
        
        # Conversation state
        self.conversation_history = []
        self.session_id = f"session_{int(time.time())}"
        self.shutdown_requested = False
        
        # Generation parameters
        self.generation_params = self.config.get('generation', {})
        
        # Auto-evolution settings
        self.auto_evolve = self.config.get('interface', {}).get('auto_evolve', True)
        self.evolution_interval = 50  # Evolve every 50 interactions
        self.interaction_count = 0
        
        # Dream during idle
        self.dream_during_idle = self.config.get('interface', {}).get('dream_during_idle', True)
        self.idle_dream_thread = None
        self.last_interaction_time = time.time()
    
    def run_console_interface(self):
        """Run console-based interactive interface"""
        print("\n" + "="*60)
        print("🧠 SAM Interactive Console")
        print("="*60)
        print("Commands:")
        print("  'exit' or 'quit' - Exit the session")
        print("  'evolve' - Trigger model evolution")
        print("  'dream' - Trigger a dream cycle")
        print("  'stats' - Show model statistics")
        print("  'save' - Save current model state")
        print("  'reset' - Reset conversation history")
        print("="*60)
        
        # Start idle dreaming thread
        if self.dream_during_idle:
            self._start_idle_dreaming()
        
        try:
            while not self.shutdown_requested:
                try:
                    # Get user input
                    user_input = input("\n🧑 You: ").strip()
                    self.last_interaction_time = time.time()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    elif user_input.lower() == 'evolve':
                        self._handle_evolve_command()
                        continue
                    elif user_input.lower() == 'dream':
                        self._handle_dream_command()
                        continue
                    elif user_input.lower() == 'stats':
                        self._handle_stats_command()
                        continue
                    elif user_input.lower() == 'save':
                        self._handle_save_command()
                        continue
                    elif user_input.lower() == 'reset':
                        self._handle_reset_command()
                        continue
                    
                    # Generate response
                    response = self._generate_response(user_input)
                    print(f"\n🧠 SAM: {response}")
                    
                    # Save conversation
                    if self.save_conversations:
                        self._save_interaction(user_input, response)
                    
                    # Auto-evolution
                    self.interaction_count += 1
                    if (self.auto_evolve and 
                        self.interaction_count % self.evolution_interval == 0):
                        print("\n🔄 Auto-evolving model...")
                        self._handle_evolve_command()
                
                except KeyboardInterrupt:
                    print("\n\nGraceful shutdown initiated...")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    logger.error(f"Interactive error: {e}")
        
        finally:
            self._cleanup()
        
        print("\n👋 Goodbye!")
        return 0
    
    def run_web_interface(self, host="localhost", port=8080):
        """Run web-based interface"""
        try:
            from flask import Flask, render_template, request, jsonify, session
            import uuid
        except ImportError:
            logger.error("Flask not installed. Install with: pip install flask")
            return 1
        
        app = Flask(__name__)
        app.secret_key = str(uuid.uuid4())
        
        @app.route('/')
        def index():
            return render_template('sam_interface.html')
        
        @app.route('/chat', methods=['POST'])
        def chat():
            data = request.json
            user_input = data.get('message', '').strip()
            
            if not user_input:
                return jsonify({'error': 'Empty message'})
            
            try:
                # Generate response
                response = self._generate_response(user_input)
                
                # Save conversation
                if self.save_conversations:
                    self._save_interaction(user_input, response)
                
                return jsonify({
                    'response': response,
                    'session_id': self.session_id
                })
            
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return jsonify({'error': str(e)})
        
        @app.route('/evolve', methods=['POST'])
        def evolve():
            try:
                results = self.model.evolve()
                return jsonify({'results': results})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/stats')
        def stats():
            try:
                stats = self.model.get_stats()
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)})
        
        print(f"\n🌐 Starting SAM web interface at http://{host}:{port}")
        app.run(host=host, port=port, debug=False)
        
        return 0
    
    def _generate_response(self, user_input: str) -> str:
        """Generate response to user input"""
        try:
            response = self.model.generate(
                input_text=user_input,
                max_length=self.generation_params.get('max_length', 200),
                temperature=self.generation_params.get('temperature', 0.8),
                top_k=self.generation_params.get('top_k', 50),
                top_p=self.generation_params
