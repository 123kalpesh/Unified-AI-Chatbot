#!/usr/bin/env python3
"""
Unified setup script for the AI Chatbot
Simple, fast setup that works with minimal dependencies
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Core packages only
    packages = ['flask', 'flask-cors', 'requests']
    
    try:
        for package in packages:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ['templates']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   Created: {directory}/")
        else:
            print(f"   Exists: {directory}/")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    required_modules = ['flask', 'requests']
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def test_chatbot():
    """Test the unified chatbot"""
    print("\n🎮 Testing unified chatbot...")
    try:
        from unified_chatbot import UnifiedChatbot
        
        # Test chatbot creation
        chatbot = UnifiedChatbot(mode='enhanced')
        print("   ✅ Unified chatbot created successfully")
        
        # Test basic functionality
        result = chatbot.generate_response("Hello!")
        print(f"   ✅ Basic response: {result['response']}")
        print(f"   ✅ Response time: {result['response_time']:.2f}s")
        print(f"   ✅ Intent detected: {result['intent']}")
        print(f"   ✅ Mode: {result['mode']}")
        
        # Test stats
        stats = chatbot.get_stats()
        print(f"   ✅ Stats: {stats['total_messages']} messages")
        
        print("✅ Unified chatbot test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Unified chatbot test failed: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n🚀 Usage Instructions:")
    print("=" * 50)
    print("1. Command Line Interface:")
    print("   python unified_chatbot.py")
    print()
    print("2. Web Interface:")
    print("   python unified_web_interface.py")
    print("   Then open: http://localhost:5000")
    print()
    print("3. Available Modes:")
    print("   • Enhanced: AI API + Rule-based responses")
    print("   • Simple: Rule-based responses only")
    print("   • TensorFlow: ML model (requires TensorFlow)")
    print()
    print("4. Features:")
    print("   • Natural Language Processing")
    print("   • Sentiment Analysis")
    print("   • Intent Detection")
    print("   • Keyword Extraction")
    print("   • Context Awareness")
    print("   • Multiple Response Modes")
    print("   • Beautiful Web Interface")
    print("   • Real-time Analytics")

def main():
    """Main setup function"""
    print("🚀 Unified AI Chatbot Setup")
    print("=" * 50)
    print("This setup installs a unified chatbot with multiple modes")
    print("and minimal dependencies for maximum compatibility.")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed during import testing")
        sys.exit(1)
    
    # Test chatbot
    if not test_chatbot():
        print("\n⚠️  Setup completed but chatbot test failed")
        print("   You can still try running the chatbot manually")
    else:
        print("\n🎉 Unified setup completed successfully!")
    
    # Show usage instructions
    show_usage_instructions()
    
    # Show next steps
    print("\n📋 Next Steps:")
    print("   1. Run 'python unified_chatbot.py' for command-line interface")
    print("   2. Run 'python unified_web_interface.py' for web interface")
    print("   3. Open browser to http://localhost:5000 for web chat")
    print("   4. Try different modes: simple, enhanced, tensorflow")
    
    print(f"\n💡 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Architecture: {platform.machine()}")
    
    print("\n🎯 Features Available:")
    print("   • Multiple chatbot modes in one file")
    print("   • Beautiful, responsive web interface")
    print("   • AI API integration (Gemini)")
    print("   • Advanced NLP processing")
    print("   • Real-time sentiment analysis")
    print("   • Intent detection and keyword extraction")
    print("   • Context-aware conversations")
    print("   • Easy mode switching")
    print("   • Minimal dependencies")
    print("   • Cross-platform compatibility")

if __name__ == "__main__":
    main()
