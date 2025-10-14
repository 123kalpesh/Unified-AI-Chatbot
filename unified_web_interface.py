#!/usr/bin/env python3
"""
Unified Web Interface for AI Chatbot
Combines all web interface features into a single, simple file
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from unified_chatbot import UnifiedChatbot
import time

app = Flask(__name__)
CORS(app)

# Initialize unified chatbot with enhanced mode
chatbot = UnifiedChatbot(mode='enhanced')

# Set API key automatically
GEMINI_API_KEY = "AIzaSyAimSGj9grDKDlDRLpN2OotJYc7nGjUKTE"
chatbot.set_api_key(GEMINI_API_KEY)

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('unified_index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with unified features"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Generate response
        result = chatbot.generate_response(user_message)
        
        return jsonify({
            'response': result['response'],
            'intent': result['intent'],
            'sentiment': result['sentiment'],
            'keywords': result['keywords'],
            'response_time': result['response_time'],
            'response_source': result['response_source'],
            'message_count': result['message_count'],
            'mode': result['mode']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/context', methods=['GET'])
def get_context():
    """Get conversation context"""
    try:
        context = chatbot.get_conversation_context()
        return jsonify({'context': context})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_context():
    """Clear conversation context"""
    try:
        chatbot.conversation_history = []
        chatbot.message_count = 0
        return jsonify({'message': 'Context cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text with NLP"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Empty text'}), 400
        
        analysis = {
            'intent': chatbot.nlp.detect_intent(text),
            'sentiment': chatbot.nlp.calculate_sentiment(text),
            'keywords': chatbot.nlp.extract_keywords(text, top_k=10),
            'cleaned_text': chatbot.nlp.clean_text(text)
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get chatbot statistics"""
    try:
        stats = chatbot.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mode', methods=['POST'])
def set_mode():
    """Change chatbot mode"""
    try:
        data = request.get_json()
        mode = data.get('mode', '').strip()
        
        if not mode:
            return jsonify({'error': 'Mode required'}), 400
        
        if chatbot.set_mode(mode):
            return jsonify({'message': f'Mode changed to {mode}', 'mode': mode})
        else:
            # Handle case where TensorFlow is not available
            if mode == 'tensorflow':
                return jsonify({
                    'message': 'TensorFlow mode not available. Using enhanced mode instead.',
                    'mode': 'enhanced',
                    'warning': 'TensorFlow is not installed. Please install it to use TensorFlow mode.'
                })
            else:
                return jsonify({'error': f'Failed to change mode to {mode}'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api_key', methods=['POST'])
def set_api_key():
    """Set API key"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 400
        
        chatbot.set_api_key(api_key)
        return jsonify({'message': 'API key set successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get chatbot status"""
    return jsonify({
        'mode': chatbot.mode,
        'api_enabled': chatbot.nlp.use_api,
        'conversation_length': len(chatbot.conversation_history),
        'message_count': chatbot.message_count,
        'response_time': chatbot.response_time
    })

if __name__ == '__main__':
    print("🌐 Starting Unified Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔑 API key is pre-configured for enhanced responses")
    print("🎯 Available modes: simple, enhanced, tensorflow")
    app.run(debug=True, host='0.0.0.0', port=5000)
