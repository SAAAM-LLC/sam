# interactive.py - SAM Interactive Interface
import sys
import json
import time
import threading
import os
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
        print("  'help' - Show this help message")
        print("  'config' - Show current configuration")
        print("  'history' - Show conversation history")
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
                    elif user_input.lower() == 'help':
                        self._handle_help_command()
                        continue
                    elif user_input.lower() == 'config':
                        self._handle_config_command()
                        continue
                    elif user_input.lower() == 'history':
                        self._handle_history_command()
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
            return self._get_web_template()
        
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
        
        @app.route('/dream', methods=['POST'])
        def dream():
            try:
                results = self.model.dreaming.dream_cycle(duration_minutes=0.1)
                return jsonify({'results': results})
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @app.route('/history')
        def history():
            try:
                return jsonify({'history': self.conversation_history[-20:]})  # Last 20 interactions
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
                top_p=self.generation_params.get('top_p', 0.9),
                private_context=False,
                use_hive_mind=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"I apologize, but I encountered an error: {e}"
    
    def _handle_evolve_command(self):
        """Handle evolution command"""
        try:
            print("🔄 Evolving SAM...")
            results = self.model.evolve()
            print(f"✅ Evolution complete:")
            
            if isinstance(results, dict):
                for key, value in results.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {results}")
                
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
    
    def _handle_dream_command(self):
        """Handle dream command"""
        try:
            print("💭 Starting dream cycle...")
            dream_results = self.model.dreaming.dream_cycle(duration_minutes=0.1)
            print(f"✅ Dream cycle complete:")
            print(f"  Dreams generated: {dream_results.get('dreams_generated', 0)}")
            print(f"  Duration: {dream_results.get('duration', 0):.2f}s")
            print(f"  New concepts: {dream_results.get('new_concepts', 0)}")
        except Exception as e:
            print(f"❌ Dream cycle failed: {e}")
    
    def _handle_stats_command(self):
        """Handle stats command"""
        try:
            stats = self.model.get_stats()
            print("\n📊 SAM Statistics:")
            print(f"  Global Step: {stats['global_step']}")
            print(f"  Model Dimension: {stats['model_dim']}")
            print(f"  Layers: {stats['num_layers']}")
            print(f"  Parameters: {stats['total_parameters']:,}")
            print(f"  Concepts: {stats['concepts']['total_concepts']}")
            print(f"  Growth Events: {stats['growth_events']}")
            
            # Consciousness state
            consciousness = self.model.consciousness.get_consciousness_summary()
            print(f"  Consciousness Level: {consciousness.get('level', 'unknown')}")
            print(f"  Consciousness Score: {consciousness.get('consciousness_score', 0):.3f}")
            
            # Session stats
            print(f"\n📈 Session Statistics:")
            print(f"  Interactions: {self.interaction_count}")
            print(f"  Session Duration: {(time.time() - float(self.session_id.split('_')[1]))/60:.1f} minutes")
            print(f"  Auto-evolution: {'Enabled' if self.auto_evolve else 'Disabled'}")
            
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
    
    def _handle_save_command(self):
        """Handle save command"""
        try:
            print("💾 Saving model...")
            save_path = self.model.save()
            print(f"✅ Model saved to: {save_path}")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def _handle_reset_command(self):
        """Handle reset command"""
        self.conversation_history = []
        self.interaction_count = 0
        print("🔄 Conversation history reset")
    
    def _handle_help_command(self):
        """Handle help command"""
        print("\n📚 SAM Interactive Help:")
        print("  Basic Commands:")
        print("    exit/quit - Exit the session")
        print("    help - Show this help message")
        print("    stats - Show detailed model statistics")
        print("    history - Show recent conversation")
        print("    config - Show current configuration")
        print("    reset - Clear conversation history")
        print("\n  Model Commands:")
        print("    evolve - Trigger model evolution")
        print("    dream - Start a dream cycle")
        print("    save - Save current model state")
        print("\n  Features:")
        print(f"    Auto-evolution: {'Enabled' if self.auto_evolve else 'Disabled'} (every {self.evolution_interval} interactions)")
        print(f"    Idle dreaming: {'Enabled' if self.dream_during_idle else 'Disabled'}")
        print(f"    Conversation saving: {'Enabled' if self.save_conversations else 'Disabled'}")
    
    def _handle_config_command(self):
        """Handle config command"""
        print("\n⚙️ Current Configuration:")
        print("  Generation Parameters:")
        for key, value in self.generation_params.items():
            print(f"    {key}: {value}")
        
        print("  Interface Settings:")
        interface_config = self.config.get('interface', {})
        for key, value in interface_config.items():
            print(f"    {key}: {value}")
        
        print("  Model Configuration:")
        model_config = vars(self.model.config)
        important_keys = ['initial_hidden_dim', 'initial_num_layers', 'neurochemical_enabled', 
                         'biological_computing', 'emergent_representations']
        for key in important_keys:
            if key in model_config:
                print(f"    {key}: {model_config[key]}")
    
    def _handle_history_command(self):
        """Handle history command"""
        if not self.conversation_history:
            print("📝 No conversation history yet")
            return
        
        print(f"\n📝 Recent Conversation History (last {min(10, len(self.conversation_history))} interactions):")
        recent_history = self.conversation_history[-10:]
        
        for i, interaction in enumerate(recent_history, 1):
            timestamp = interaction.get('timestamp', 0)
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            user_input = interaction.get('user_input', '')[:50]
            sam_response = interaction.get('sam_response', '')[:50]
            
            print(f"  {i}. [{time_str}]")
            print(f"     You: {user_input}{'...' if len(interaction.get('user_input', '')) > 50 else ''}")
            print(f"     SAM: {sam_response}{'...' if len(interaction.get('sam_response', '')) > 50 else ''}")
    
    def _save_interaction(self, user_input: str, response: str):
        """Save interaction to conversation history"""
        interaction = {
            'timestamp': time.time(),
            'user_input': user_input,
            'sam_response': response,
            'session_id': self.session_id,
            'interaction_number': self.interaction_count
        }
        
        self.conversation_history.append(interaction)
        
        # Limit history size
        max_history = self.config.get('interface', {}).get('conversation_history', 100)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        # Save to file periodically
        if len(self.conversation_history) % 10 == 0:
            self._save_conversation_to_file()
    
    def _save_conversation_to_file(self):
        """Save conversation history to file"""
        try:
            os.makedirs("logs/conversations", exist_ok=True)
            filename = f"logs/conversations/{self.session_id}.json"
            
            conversation_data = {
                'session_id': self.session_id,
                'start_time': float(self.session_id.split('_')[1]),
                'total_interactions': self.interaction_count,
                'auto_evolve_enabled': self.auto_evolve,
                'conversation_history': self.conversation_history
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def _start_idle_dreaming(self):
        """Start background thread for idle dreaming"""
        def dream_loop():
            while not self.shutdown_requested:
                time.sleep(30)  # Check every 30 seconds
                
                # If idle for more than 2 minutes, start dreaming
                if (time.time() - self.last_interaction_time > 120 and 
                    not self.shutdown_requested):
                    try:
                        logger.info("Starting idle dream cycle...")
                        self.model.dreaming.dream_cycle(duration_minutes=0.05)
                    except Exception as e:
                        logger.error(f"Idle dreaming error: {e}")
        
        self.idle_dream_thread = threading.Thread(target=dream_loop, daemon=True)
        self.idle_dream_thread.start()
        logger.info("Idle dreaming thread started")
    
    def _cleanup(self):
        """Cleanup resources"""
        self.shutdown_requested = True
        
        # Save final conversation
        if self.save_conversations and self.conversation_history:
            self._save_conversation_to_file()
        
        # Wait for idle dream thread
        if self.idle_dream_thread and self.idle_dream_thread.is_alive():
            self.idle_dream_thread.join(timeout=5)
    
    def _get_web_template(self):
        """Get HTML template for web interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM Interactive Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1rem;
            text-align: center;
            color: white;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.8;
        }
        
        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.05);
            margin: 1rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            max-width: 80%;
            padding: 1rem;
            border-radius: 15px;
            word-wrap: break-word;
        }
        
        .user-message {
            align-self: flex-end;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }
        
        .sam-message {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
        }
        
        .input-container {
            padding: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            gap: 1rem;
        }
        
        .input-container input {
            flex: 1;
            padding: 1rem;
            border: none;
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.9);
            font-size: 1rem;
            outline: none;
        }
        
        .input-container button {
            padding: 1rem 2rem;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .input-container button:hover {
            transform: scale(1.05);
        }
        
        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            margin: 1rem 1rem 1rem 0;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .sidebar h3 {
            color: white;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .control-button {
            padding: 0.75rem;
            border: none;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .control-button:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .stats-display {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            color: white;
            font-size: 0.9rem;
        }
        
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            
            .sidebar {
                width: auto;
                margin: 0 1rem 1rem 1rem;
                order: -1;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 SAM Interactive Interface</h1>
        <p>Synergistic Autonomous Machine - Web Interface</p>
    </div>
    
    <div class="container">
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message sam-message">
                    <strong>SAM:</strong> Hello! I'm SAM, a Synergistic Autonomous Machine. I can learn, evolve, and grow through our conversation. How can I help you today?
                </div>
            </div>
            
            <div class="input-container">
                <input type="text" id="messageInput" placeholder="Type your message here..." maxlength="1000">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="sidebar">
            <h3>Controls</h3>
            <button class="control-button" onclick="evolveModel()">🔄 Evolve Model</button>
            <button class="control-button" onclick="dreamCycle()">💭 Dream Cycle</button>
            <button class="control-button" onclick="showStats()">📊 Show Stats</button>
            <button class="control-button" onclick="showHistory()">📝 Show History</button>
            <button class="control-button" onclick="clearChat()">🗑️ Clear Chat</button>
            
            <div class="stats-display" id="statsDisplay">
                <strong>Session Info:</strong><br>
                Session started: <span id="sessionTime"></span><br>
                Messages: <span id="messageCount">0</span>
            </div>
        </div>
    </div>

    <script>
        let messageCount = 0;
        const sessionStartTime = new Date();
        
        document.getElementById('sessionTime').textContent = sessionStartTime.toLocaleTimeString();
        
        // Send message on Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        function addMessage(content, isUser = false) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'sam-message'}`;
            messageDiv.innerHTML = `<strong>${isUser ? 'You' : 'SAM'}:</strong> ${content}`;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            if (isUser) {
                messageCount++;
                document.getElementById('messageCount').textContent = messageCount;
            }
        }
        
        function setLoading(loading) {
            document.body.classList.toggle('loading', loading);
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            setLoading(true);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage(`Error: ${data.error}`, false);
                } else {
                    addMessage(data.response, false);
                }
            } catch (error) {
                addMessage(`Network error: ${error.message}`, false);
            } finally {
                setLoading(false);
            }
        }
        
        async function evolveModel() {
            setLoading(true);
            addMessage('🔄 Triggering model evolution...', false);
            
            try {
                const response = await fetch('/evolve', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    addMessage(`Evolution error: ${data.error}`, false);
                } else {
                    addMessage(`✅ Evolution complete: ${JSON.stringify(data.results, null, 2)}`, false);
                }
            } catch (error) {
                addMessage(`Evolution network error: ${error.message}`, false);
            } finally {
                setLoading(false);
            }
        }
        
        async function dreamCycle() {
            setLoading(true);
            addMessage('💭 Starting dream cycle...', false);
            
            try {
                const response = await fetch('/dream', { method: 'POST' });
                const data = await response.json();
                
                if (data.error) {
                    addMessage(`Dream error: ${data.error}`, false);
                } else {
                    addMessage(`✅ Dream cycle complete: ${JSON.stringify(data.results, null, 2)}`, false);
                }
            } catch (error) {
                addMessage(`Dream network error: ${error.message}`, false);
            } finally {
                setLoading(false);
            }
        }
        
        async function showStats() {
            setLoading(true);
            
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                if (data.error) {
                    addMessage(`Stats error: ${data.error}`, false);
                } else {
                    const statsText = `📊 SAM Statistics:
                    
Global Step: ${data.global_step || 0}
Model Dimension: ${data.model_dim || 'Unknown'}
Layers: ${data.num_layers || 0}
Parameters: ${(data.total_parameters || 0).toLocaleString()}
Concepts: ${data.concepts?.total_concepts || 0}
Growth Events: ${data.growth_events || 0}
Consciousness Level: ${data.consciousness?.level || 'Unknown'}
Consciousness Score: ${(data.consciousness?.consciousness_score || 0).toFixed(3)}`;
                    
                    addMessage(statsText, false);
                }
            } catch (error) {
                addMessage(`Stats network error: ${error.message}`, false);
            } finally {
                setLoading(false);
            }
        }
        
        async function showHistory() {
            setLoading(true);
            
            try {
                const response = await fetch('/history');
                const data = await response.json();
                
                if (data.error) {
                    addMessage(`History error: ${data.error}`, false);
                } else {
                    const history = data.history || [];
                    if (history.length === 0) {
                        addMessage('📝 No conversation history found.', false);
                    } else {
                        let historyText = '📝 Recent conversation history:\\n\\n';
                        history.slice(-5).forEach((interaction, index) => {
                            const time = new Date(interaction.timestamp * 1000).toLocaleTimeString();
                            historyText += `${index + 1}. [${time}]\\n`;
                            historyText += `   You: ${interaction.user_input.substring(0, 50)}${interaction.user_input.length > 50 ? '...' : ''}\\n`;
                            historyText += `   SAM: ${interaction.sam_response.substring(0, 50)}${interaction.sam_response.length > 50 ? '...' : ''}\\n\\n`;
                        });
                        addMessage(historyText, false);
                    }
                }
            } catch (error) {
                addMessage(`History network error: ${error.message}`, false);
            } finally {
                setLoading(false);
            }
        }
        
        function clearChat() {
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.innerHTML = '<div class="message sam-message"><strong>SAM:</strong> Chat cleared. How can I help you?</div>';
            messageCount = 0;
            document.getElementById('messageCount').textContent = messageCount;
        }
    </script>
</body>
</html>
        """
    
    def request_shutdown(self):
        """Request graceful shutdown"""
        self.shutdown_requested = True

# Additional utility functions for interactive mode
def load_interactive_config(config_path="configs/interact_config.json"):
    """Load interactive configuration from file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Return default configuration
        return {
            "generation": {
                "max_length": 200,
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "interface": {
                "save_conversations": True,
                "conversation_history": 100,
                "auto_evolve": True,
                "dream_during_idle": True
            },
            "privacy": {
                "enable_private_context": True,
                "log_interactions": False,
                "anonymize_data": False
            }
        }

def create_interactive_session(model_path=None, config_path=None, web_mode=False, host="localhost", port=8080):
    """Create and run an interactive session with SAM"""
    try:
        # Import SAM
        from sam import SAM, create_sam_model
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = SAM.load(model_path)
        else:
            print("Creating new SAM model...")
            model, _ = create_sam_model()
        
        # Load configuration
        config = load_interactive_config(config_path)
        
        # Create interactive interface
        interactive = SAMInteractive(
            model=model,
            config=config,
            save_conversations=config.get('interface', {}).get('save_conversations', True)
        )
        
        # Run interface
        if web_mode:
            return interactive.run_web_interface(host=host, port=port)
        else:
            return interactive.run_console_interface()
            
    except Exception as e:
        print(f"Failed to start interactive session: {e}")
        import traceback
        traceback.print_exc()
        return 1

def batch_conversation(model, conversations_file, output_file=None):
    """Process a batch of conversations from a file"""
    try:
        with open(conversations_file, 'r') as f:
            conversations = json.load(f)
        
        results = []
        interactive = SAMInteractive(model, save_conversations=False)
        
        for i, conversation in enumerate(conversations):
            print(f"Processing conversation {i+1}/{len(conversations)}")
            
            if isinstance(conversation, str):
                # Single message
                response = interactive._generate_response(conversation)
                results.append({
                    'input': conversation,
                    'output': response,
                    'conversation_id': i
                })
            elif isinstance(conversation, dict):
                # Conversation with metadata
                input_text = conversation.get('input', conversation.get('message', ''))
                response = interactive._generate_response(input_text)
                
                result = {
                    'input': input_text,
                    'output': response,
                    'conversation_id': i,
                    'metadata': conversation.get('metadata', {})
                }
                results.append(result)
            elif isinstance(conversation, list):
                # Multi-turn conversation
                conversation_results = []
                for turn in conversation:
                    if isinstance(turn, str):
                        response = interactive._generate_response(turn)
                        conversation_results.append({
                            'input': turn,
                            'output': response
                        })
                    elif isinstance(turn, dict):
                        input_text = turn.get('input', turn.get('message', ''))
                        response = interactive._generate_response(input_text)
                        conversation_results.append({
                            'input': input_text,
                            'output': response,
                            'metadata': turn.get('metadata', {})
                        })
                
                results.append({
                    'conversation_id': i,
                    'turns': conversation_results
                })
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None

def evaluate_conversation_quality(conversations_file, model):
    """Evaluate the quality of conversations"""
    try:
        with open(conversations_file, 'r') as f:
            conversations = json.load(f)
        
        metrics = {
            'total_conversations': len(conversations),
            'avg_response_length': 0,
            'coherence_scores': [],
            'diversity_scores': [],
            'engagement_scores': []
        }
        
        interactive = SAMInteractive(model, save_conversations=False)
        total_length = 0
        
        for conversation in conversations:
            if isinstance(conversation, dict) and 'turns' in conversation:
                # Multi-turn conversation
                for turn in conversation['turns']:
                    response = turn.get('output', '')
                    total_length += len(response)
                    
                    # Calculate metrics
                    coherence = calculate_coherence_score(turn.get('input', ''), response)
                    diversity = calculate_diversity_score(response)
                    engagement = calculate_engagement_score(response)
                    
                    metrics['coherence_scores'].append(coherence)
                    metrics['diversity_scores'].append(diversity)
                    metrics['engagement_scores'].append(engagement)
            else:
                # Single conversation
                response = conversation.get('output', '')
                total_length += len(response)
                
                coherence = calculate_coherence_score(conversation.get('input', ''), response)
                diversity = calculate_diversity_score(response)
                engagement = calculate_engagement_score(response)
                
                metrics['coherence_scores'].append(coherence)
                metrics['diversity_scores'].append(diversity)
                metrics['engagement_scores'].append(engagement)
        
        # Calculate averages
        if metrics['coherence_scores']:
            metrics['avg_response_length'] = total_length / len(metrics['coherence_scores'])
            metrics['avg_coherence'] = sum(metrics['coherence_scores']) / len(metrics['coherence_scores'])
            metrics['avg_diversity'] = sum(metrics['diversity_scores']) / len(metrics['diversity_scores'])
            metrics['avg_engagement'] = sum(metrics['engagement_scores']) / len(metrics['engagement_scores'])
        
        return metrics
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

def calculate_coherence_score(input_text, response):
    """Calculate coherence score between input and response"""
    # Simple coherence metric based on word overlap and response appropriateness
    input_words = set(input_text.lower().split())
    response_words = set(response.lower().split())
    
    if not input_words or not response_words:
        return 0.0
    
    # Word overlap
    overlap = len(input_words.intersection(response_words))
    overlap_score = overlap / len(input_words.union(response_words))
    
    # Response length appropriateness
    length_ratio = len(response) / max(len(input_text), 1)
    length_score = min(1.0, length_ratio / 3.0)  # Optimal ratio around 2-3x
    
    # Check for question-answer patterns
    is_question = '?' in input_text
    has_answer_words = any(word in response.lower() for word in ['yes', 'no', 'because', 'since', 'therefore'])
    qa_score = 1.0 if (is_question and has_answer_words) else 0.5
    
    # Combine scores
    coherence = (overlap_score * 0.4 + length_score * 0.3 + qa_score * 0.3)
    return min(1.0, coherence)

def calculate_diversity_score(text):
    """Calculate lexical diversity of text"""
    if not text:
        return 0.0
    
    words = text.split()
    if len(words) < 2:
        return 0.0
    
    unique_words = len(set(words))
    diversity = unique_words / len(words)
    
    return diversity

def calculate_engagement_score(text):
    """Calculate engagement score based on text features"""
    if not text:
        return 0.0
    
    engagement_indicators = {
        'questions': text.count('?'),
        'exclamations': text.count('!'),
        'personal_pronouns': sum(1 for word in ['you', 'your', 'we', 'us', 'our'] if word in text.lower()),
        'emotional_words': sum(1 for word in ['feel', 'think', 'believe', 'hope', 'love', 'hate', 'excited', 'amazing'] if word in text.lower()),
        'conversational_markers': sum(1 for phrase in ['by the way', 'speaking of', 'that reminds me', 'actually'] if phrase in text.lower())
    }
    
    # Normalize scores
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    
    normalized_score = sum(engagement_indicators.values()) / word_count
    engagement = min(1.0, normalized_score * 10)  # Scale up and cap at 1.0
    
    return engagement

def conversation_analytics(conversations_file):
    """Analyze conversation patterns and provide insights"""
    try:
        with open(conversations_file, 'r') as f:
            conversations = json.load(f)
        
        analytics = {
            'total_conversations': len(conversations),
            'conversation_lengths': [],
            'most_common_topics': [],
            'response_times': [],
            'user_engagement_patterns': {},
            'peak_usage_times': [],
            'conversation_flows': []
        }
        
        # Analyze each conversation
        for conversation in conversations:
            if isinstance(conversation, dict):
                if 'turns' in conversation:
                    # Multi-turn conversation
                    analytics['conversation_lengths'].append(len(conversation['turns']))
                    
                    # Analyze conversation flow
                    flow = []
                    for turn in conversation['turns']:
                        input_length = len(turn.get('input', '').split())
                        output_length = len(turn.get('output', '').split())
                        flow.append({'input_length': input_length, 'output_length': output_length})
                    
                    analytics['conversation_flows'].append(flow)
                    
                    # Extract timestamps if available
                    if 'timestamp' in conversation:
                        import datetime
                        dt = datetime.datetime.fromtimestamp(conversation['timestamp'])
                        analytics['peak_usage_times'].append(dt.hour)
                
                else:
                    # Single conversation
                    analytics['conversation_lengths'].append(1)
                    
                    # Extract topics (simple keyword extraction)
                    input_text = conversation.get('input', '').lower()
                    topics = extract_topics(input_text)
                    analytics['most_common_topics'].extend(topics)
        
        # Calculate statistics
        if analytics['conversation_lengths']:
            analytics['avg_conversation_length'] = sum(analytics['conversation_lengths']) / len(analytics['conversation_lengths'])
            analytics['max_conversation_length'] = max(analytics['conversation_lengths'])
            analytics['min_conversation_length'] = min(analytics['conversation_lengths'])
        
        # Find most common topics
        from collections import Counter
        topic_counts = Counter(analytics['most_common_topics'])
        analytics['top_topics'] = topic_counts.most_common(10)
        
        # Find peak usage hours
        if analytics['peak_usage_times']:
            hour_counts = Counter(analytics['peak_usage_times'])
            analytics['peak_hours'] = hour_counts.most_common(5)
        
        return analytics
        
    except Exception as e:
        print(f"Analytics failed: {e}")
        return None

def extract_topics(text):
    """Extract simple topics from text"""
    # Simple keyword-based topic extraction
    topic_keywords = {
        'technology': ['computer', 'software', 'ai', 'robot', 'digital', 'internet'],
        'science': ['research', 'experiment', 'study', 'theory', 'analysis', 'data'],
        'education': ['learn', 'teach', 'school', 'university', 'student', 'knowledge'],
        'health': ['medical', 'doctor', 'health', 'medicine', 'treatment', 'therapy'],
        'business': ['company', 'business', 'market', 'finance', 'money', 'economy'],
        'entertainment': ['movie', 'music', 'game', 'fun', 'entertainment', 'hobby'],
        'personal': ['family', 'friend', 'relationship', 'personal', 'life', 'feel']
    }
    
    topics = []
    words = text.split()
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in words for keyword in keywords):
            topics.append(topic)
    
    return topics

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM Interactive Interface")
    parser.add_argument("--model", type=str, help="Path to SAM model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--web", action="store_true", help="Start web interface")
    parser.add_argument("--host", type=str, default="localhost", help="Web interface host")
    parser.add_argument("--port", type=int, default=8080, help="Web interface port")
    parser.add_argument("--batch", type=str, help="Process batch conversations from file")
    parser.add_argument("--output", type=str, help="Output file for batch processing")
    parser.add_argument("--evaluate", type=str, help="Evaluate conversation quality from file")
    parser.add_argument("--analytics", type=str, help="Run analytics on conversation file")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        from sam import SAM, create_sam_model
        
        if args.model:
            model = SAM.load(args.model)
        else:
            model, _ = create_sam_model()
        
        results = batch_conversation(model, args.batch, args.output)
        if results:
            print(f"Processed {len(results)} conversations")
    
    elif args.evaluate:
        # Evaluation mode
        from sam import SAM, create_sam_model
        
        if args.model:
            model = SAM.load(args.model)
        else:
            model, _ = create_sam_model()
        
        metrics = evaluate_conversation_quality(args.evaluate, model)
        if metrics:
            print("Conversation Quality Metrics:")
            for key, value in metrics.items():
                if isinstance(value, list):
                    continue  # Skip list details
                print(f"  {key}: {value}")
    
    elif args.analytics:
        # Analytics mode
        analytics = conversation_analytics(args.analytics)
        if analytics:
            print("Conversation Analytics:")
            for key, value in analytics.items():
                if key in ['conversation_flows', 'peak_usage_times']:
                    continue  # Skip complex data
                print(f"  {key}: {value}")
    
    else:
        # Interactive mode
        exit_code = create_interactive_session(
            model_path=args.model,
            config_path=args.config,
            web_mode=args.web,
            host=args.host,
            port=args.port
        )
        exit(exit_code)
