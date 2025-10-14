#!/usr/bin/env python3
"""
Unified AI Chatbot - Combines all chatbot features into a single, simple file
Supports multiple modes: Simple, Enhanced, and TensorFlow-based
"""

import re
import random
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math

# Try to import TensorFlow, but don't fail if it's not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input, Concatenate
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import numpy as np
    import pickle
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. Using simple mode only.")

class UnifiedNLPProcessor:
    """Unified NLP processor that works with or without external dependencies"""
    
    def __init__(self):
        # Basic stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'this',
            'these', 'those', 'have', 'had', 'do', 'does', 'did', 'can',
            'could', 'would', 'should', 'may', 'might', 'must'
        }
        
        # Enhanced sentiment words
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'perfect', 'love', 'like', 'enjoy', 'happy',
            'pleased', 'satisfied', 'delighted', 'impressed', 'outstanding',
            'best', 'better', 'nice', 'cool', 'super', 'marvelous', 'incredible',
            'fabulous', 'magnificent', 'spectacular', 'outstanding', 'exceptional'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate',
            'dislike', 'angry', 'sad', 'disappointed', 'frustrated', 'annoyed',
            'upset', 'worried', 'concerned', 'problem', 'issue', 'error',
            'worst', 'worse', 'horrible', 'terrible', 'awful', 'disgusting',
            'frustrating', 'annoying', 'disappointing', 'upsetting'
        }
        
        # API configuration
        self.api_key = "AIzaSyAimSGj9grDKDlDRLpN2OotJYc7nGjUKTE"  # Gemini API key
        self.use_api = False
        
    def set_api_key(self, api_key: str):
        """Set API key for enhanced responses"""
        self.api_key = api_key
        self.use_api = True
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def calculate_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate enhanced sentiment scores"""
        tokens = self.tokenize(self.clean_text(text))
        
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        total_words = len(tokens)
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Enhanced sentiment calculation with intensity
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        # Boost scores for strong words
        strong_positive = sum(1 for token in tokens if token in ['love', 'amazing', 'fantastic', 'brilliant'])
        strong_negative = sum(1 for token in tokens if token in ['hate', 'terrible', 'awful', 'horrible'])
        
        positive_score += strong_positive * 0.1
        negative_score += strong_negative * 0.1
        
        neutral_score = max(0, 1.0 - positive_score - negative_score)
        
        return {
            'positive': min(1.0, positive_score),
            'negative': min(1.0, negative_score),
            'neutral': neutral_score
        }
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords using enhanced frequency counting"""
        tokens = self.remove_stopwords(self.tokenize(self.clean_text(text)))
        word_counts = Counter(tokens)
        
        # Boost important words
        important_words = ['python', 'programming', 'ai', 'machine', 'learning', 'data', 'code']
        for word in important_words:
            if word in word_counts:
                word_counts[word] *= 2
        
        return [word for word, count in word_counts.most_common(top_k)]
    
    def detect_intent(self, text: str) -> str:
        """Enhanced intent detection"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']):
            return 'greeting'
        elif any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you explain']):
            return 'question'
        elif any(word in text_lower for word in ['help', 'assist', 'support', 'guide', 'explain', 'teach']):
            return 'help_request'
        elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate', 'grateful']):
            return 'compliment'
        elif any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'farewell', 'later']):
            return 'goodbye'
        elif any(word in text_lower for word in ['tell me about', 'explain', 'describe', 'what is']):
            return 'explanation_request'
        elif any(word in text_lower for word in ['create', 'make', 'build', 'develop', 'code']):
            return 'creation_request'
        else:
            return 'statement'
    
    def get_ai_response(self, user_input: str, context: str = "") -> Optional[str]:
        """Get response from Gemini API"""
        if not self.use_api or not self.api_key:
            return None
        
        try:
            # Use Google Gemini API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key={self.api_key}"
            
            prompt = f"""You are a helpful AI assistant. Respond naturally and conversationally.
            
Context: {context}
User: {user_input}

Respond as a friendly AI assistant:"""
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 150,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            response = requests.post(
                url,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
                else:
                    print(f"Gemini API Error: No candidates in response")
                    return None
            else:
                print(f"Gemini API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return None

class UnifiedChatbot:
    """Unified AI Chatbot with multiple modes and features"""
    
    def __init__(self, mode='enhanced'):
        """
        Initialize chatbot with specified mode
        Modes: 'simple', 'enhanced', 'tensorflow'
        """
        self.mode = mode
        self.nlp = UnifiedNLPProcessor()
        self.conversation_history = []
        self.context_window = 10
        self.response_time = 0
        self.message_count = 0
        
        # Initialize TensorFlow model if available and requested
        if mode == 'tensorflow' and TENSORFLOW_AVAILABLE:
            self._init_tensorflow_model()
        elif mode == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available, falling back to enhanced mode")
            self.mode = 'enhanced'
        
        # Enhanced response templates
        self.response_templates = {
            'greeting': [
                "Hello! 👋 I'm your AI assistant. How can I help you today?",
                "Hi there! 😊 What would you like to know?",
                "Greetings! 🌟 I'm here to assist you with anything you need.",
                "Hey! 🚀 Nice to meet you. How can I help?",
                "Hello! 💫 I'm your intelligent assistant. What can I do for you?"
            ],
            'question': [
                "That's a great question! 🤔 Let me help you with that.",
                "Excellent question! 💡 Here's what I can tell you about that.",
                "I'd be happy to explain that! 📚 Let me break it down for you.",
                "That's a good question! 🎯 Here's what I know about that topic.",
                "Great question! 🌟 Let me provide you with some insights on that."
            ],
            'help_request': [
                "I'm here to help! 🛠️ What specifically do you need assistance with?",
                "I'd be happy to assist you! 💪 What's the issue you're facing?",
                "Let me help you with that! 🤝 Can you provide more details?",
                "I'm ready to help! ⚡ What would you like to know?",
                "Sure thing! 🎯 I'm here to support you. What do you need?"
            ],
            'compliment': [
                "You're very welcome! 😊 I'm glad I could help!",
                "Thank you so much! 🙏 I'm happy to assist you!",
                "You're welcome! 🌟 It's my pleasure to help!",
                "Thanks! 💖 I'm here whenever you need me!",
                "You're welcome! 🎉 I enjoy helping you!"
            ],
            'goodbye': [
                "Goodbye! 👋 It was nice chatting with you!",
                "See you later! 🌟 Feel free to come back anytime!",
                "Take care! 💫 I'm here if you need anything else!",
                "Farewell! 🚀 It was great talking with you!",
                "Goodbye! 🎯 Have a wonderful day!"
            ],
            'explanation_request': [
                "I'd love to explain that! 📖 Let me break it down for you.",
                "Great topic! 🎓 Let me provide you with a detailed explanation.",
                "That's fascinating! 🔍 Let me share what I know about that.",
                "Excellent question! 💭 Let me give you a comprehensive answer.",
                "I'm excited to explain that! ⚡ Here's what you need to know."
            ],
            'creation_request': [
                "I'd be happy to help you create something! 🛠️ What do you have in mind?",
                "Great idea! 💡 Let me help you build that!",
                "I love creative projects! 🎨 What would you like to create?",
                "Let's build something amazing together! 🚀 What's your vision?",
                "I'm excited to help you create! ⚡ What are we building?"
            ],
            'default': [
                "That's interesting! 🤔 Tell me more about that.",
                "I see! 👀 Can you elaborate on that?",
                "That's a good point! 💭 What else would you like to know?",
                "I understand! 🎯 Is there anything specific you'd like to discuss?",
                "That's fascinating! 🌟 I'd love to learn more about your thoughts on this."
            ]
        }
        
        # Enhanced knowledge base
        self.knowledge_base = {
            'programming': {
                'keywords': ['code', 'programming', 'software', 'development', 'python', 'javascript', 'java', 'coding'],
                'responses': [
                    "Programming is amazing! 💻 What language are you working with?",
                    "I love talking about programming! 🚀 What specific topic interests you?",
                    "Programming can be challenging but very rewarding! ⚡ What would you like to know?",
                    "There are many exciting programming languages and frameworks! 🎯 What's your focus?"
                ]
            },
            'ai': {
                'keywords': ['artificial intelligence', 'ai', 'machine learning', 'neural network', 'algorithm', 'deep learning'],
                'responses': [
                    "AI is a fascinating field! 🤖 What aspect of AI interests you most?",
                    "Machine learning and AI are rapidly evolving! 🧠 What would you like to explore?",
                    "There's so much happening in AI these days! ⚡ What specific area catches your interest?",
                    "AI has many applications from chatbots to self-driving cars! 🚗 What intrigues you?"
                ]
            },
            'technology': {
                'keywords': ['technology', 'tech', 'computer', 'internet', 'software', 'hardware', 'digital'],
                'responses': [
                    "Technology is constantly evolving! 📱 What tech topic interests you?",
                    "There are so many exciting developments in technology! 🌟 What catches your eye?",
                    "Technology affects our daily lives in many ways! 💡 What aspect interests you?",
                    "From smartphones to AI, technology is everywhere! 🌐 What would you like to discuss?"
                ]
            },
            'data': {
                'keywords': ['data', 'analytics', 'database', 'big data', 'statistics', 'analysis'],
                'responses': [
                    "Data is the new oil! 📊 What kind of data analysis interests you?",
                    "Data science is incredibly powerful! 📈 What would you like to explore?",
                    "There's so much insight hidden in data! 🔍 What specific area interests you?",
                    "Data drives decisions in every industry! 💼 What aspect would you like to discuss?"
                ]
            }
        }
    
    def _init_tensorflow_model(self):
        """Initialize TensorFlow model components"""
        if not TENSORFLOW_AVAILABLE:
            return
        
        self.vocab_size = 10000
        self.max_length = 50
        self.embedding_dim = 128
        self.lstm_units = 256
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None
    
    def generate_response(self, user_input: str) -> Dict[str, any]:
        """Generate response with metadata"""
        start_time = time.time()
        
        # Clean and analyze input
        cleaned_input = self.nlp.clean_text(user_input)
        intent = self.nlp.detect_intent(user_input)
        sentiment = self.nlp.calculate_sentiment(user_input)
        keywords = self.nlp.extract_keywords(user_input, top_k=3)
        
        # Add to conversation history
        self.conversation_history.append(user_input)
        if len(self.conversation_history) > self.context_window * 2:
            self.conversation_history = self.conversation_history[-(self.context_window * 2):]
        
        # Generate response based on mode
        if self.mode == 'tensorflow' and self.model:
            response = self._generate_tensorflow_response(user_input)
            response_source = "TensorFlow Model"
        else:
            # Try AI API first
            context = self._get_recent_context()
            ai_response = self.nlp.get_ai_response(user_input, context)
            
            if ai_response:
                response = ai_response
                response_source = "AI API"
            else:
                # Fallback to rule-based response
                response = self._generate_intent_response(intent, keywords, sentiment)
                response_source = "Rule-based"
        
        # Calculate response time
        self.response_time = time.time() - start_time
        self.message_count += 1
        
        # Add response to history
        self.conversation_history.append(response)
        
        return {
            'response': response,
            'intent': intent,
            'sentiment': sentiment,
            'keywords': keywords,
            'response_time': self.response_time,
            'response_source': response_source,
            'message_count': self.message_count,
            'mode': self.mode
        }
    
    def _generate_tensorflow_response(self, user_input: str) -> str:
        """Generate response using TensorFlow model"""
        # This would use the trained model
        # For now, fallback to rule-based
        return self._generate_intent_response(
            self.nlp.detect_intent(user_input),
            self.nlp.extract_keywords(user_input),
            self.nlp.calculate_sentiment(user_input)
        )
    
    def _generate_intent_response(self, intent: str, keywords: List[str], sentiment: Dict[str, float]) -> str:
        """Generate response based on detected intent"""
        
        # Check for topic-specific responses
        topic_response = self._get_topic_response(keywords)
        if topic_response:
            return topic_response
        
        # Get base response for intent
        if intent in self.response_templates:
            base_response = random.choice(self.response_templates[intent])
        else:
            base_response = random.choice(self.response_templates['default'])
        
        # Enhance response based on sentiment
        if sentiment['positive'] > 0.3:
            if not any(word in base_response.lower() for word in ['great', 'excellent', 'wonderful']):
                base_response = f"Great! {base_response}"
        elif sentiment['negative'] > 0.3:
            base_response = f"I understand your concern. {base_response}"
        
        # Add keyword awareness
        if keywords and len(keywords) > 0:
            main_keyword = keywords[0]
            if main_keyword not in base_response.lower() and intent == 'question':
                base_response = base_response.replace("that", f"that regarding {main_keyword}")
        
        return base_response
    
    def _get_topic_response(self, keywords: List[str]) -> str:
        """Get topic-specific response based on keywords"""
        for topic, data in self.knowledge_base.items():
            if any(keyword in keywords for keyword in data['keywords']):
                return random.choice(data['responses'])
        return None
    
    def _get_recent_context(self) -> str:
        """Get recent conversation context"""
        if len(self.conversation_history) < 2:
            return ""
        
        recent_messages = self.conversation_history[-4:]  # Last 4 messages
        context = "Recent conversation:\n"
        for i, message in enumerate(recent_messages):
            role = "User" if i % 2 == 0 else "Assistant"
            context += f"{role}: {message}\n"
        
        return context
    
    def get_conversation_context(self) -> str:
        """Get current conversation context"""
        if not self.conversation_history:
            return "No conversation history."
        
        context = "Recent conversation:\n"
        for i, message in enumerate(self.conversation_history[-10:]):  # Last 10 messages
            role = "User" if i % 2 == 0 else "Assistant"
            context += f"{role}: {message}\n"
        
        return context
    
    def get_stats(self) -> Dict[str, any]:
        """Get chatbot statistics"""
        return {
            'total_messages': self.message_count,
            'conversation_length': len(self.conversation_history),
            'average_response_time': self.response_time,
            'context_window': self.context_window,
            'mode': self.mode,
            'api_enabled': self.nlp.use_api
        }
    
    def set_api_key(self, api_key: str):
        """Set API key for enhanced responses"""
        self.nlp.set_api_key(api_key)
    
    def set_mode(self, mode: str):
        """Change chatbot mode"""
        if mode == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            print("⚠️  TensorFlow not available, keeping current mode")
            return False
        
        self.mode = mode
        if mode == 'tensorflow':
            self._init_tensorflow_model()
        return True

def main():
    """Main function for command-line interface"""
    print("🤖 Unified AI Chatbot")
    print("=" * 50)
    print("Available modes: simple, enhanced, tensorflow")
    print("Commands: 'analyze', 'context', 'stats', 'mode <mode>', 'api <key>'")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("=" * 50)
    
    # Initialize with enhanced mode by default
    chatbot = UnifiedChatbot(mode='enhanced')
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 Bot: Goodbye! It was nice chatting with you. 👋")
                break
            
            elif user_input.lower() == 'analyze':
                if chatbot.conversation_history:
                    last_input = chatbot.conversation_history[-1]
                    analysis = chatbot.nlp.calculate_sentiment(last_input)
                    intent = chatbot.nlp.detect_intent(last_input)
                    keywords = chatbot.nlp.extract_keywords(last_input, top_k=5)
                    
                    print(f"\n📊 Analysis of: '{last_input}'")
                    print(f"Intent: {intent}")
                    print(f"Sentiment: {analysis}")
                    print(f"Keywords: {keywords}")
                else:
                    print("\n📊 No input to analyze yet.")
                continue
            
            elif user_input.lower() == 'context':
                context = chatbot.get_conversation_context()
                print(f"\n📝 {context}")
                continue
            
            elif user_input.lower() == 'stats':
                stats = chatbot.get_stats()
                print(f"\n📈 Chatbot Statistics:")
                print(f"Mode: {stats['mode']}")
                print(f"Total messages: {stats['total_messages']}")
                print(f"Conversation length: {stats['conversation_length']}")
                print(f"Average response time: {stats['average_response_time']:.2f}s")
                print(f"API enabled: {stats['api_enabled']}")
                continue
            
            elif user_input.lower().startswith('mode '):
                new_mode = user_input[5:].strip()
                if chatbot.set_mode(new_mode):
                    print(f"\n🔄 Mode changed to: {new_mode}")
                else:
                    print(f"\n❌ Failed to change mode to: {new_mode}")
                continue
            
            elif user_input.lower().startswith('api '):
                api_key = user_input[4:].strip()
                chatbot.set_api_key(api_key)
                print(f"\n🔑 API key set! Enhanced responses enabled.")
                continue
            
            if user_input:
                result = chatbot.generate_response(user_input)
                print(f"\n🤖 Bot: {result['response']}")
                print(f"   📊 Intent: {result['intent']} | Source: {result['response_source']} | Time: {result['response_time']:.2f}s")
            else:
                print("\n🤖 Bot: I didn't catch that. Could you please repeat?")
                
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
