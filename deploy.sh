#!/bin/bash
# SSA Conjunction Analysis Engine - Deployment Script

echo "🚀 SSA Conjunction Analysis Engine Deployment"
echo "============================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    cp .env.template .env
    echo "✅ .env file created. Please review and customize it."
    echo ""
    echo "🔧 REQUIRED STEPS:"
    echo "1. Edit .env file and set your configuration"
    echo "2. For demo mode: Leave database/password fields empty"  
    echo "3. For production: Set all credentials and passwords"
    echo "4. Run: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "📚 AVAILABLE COMMANDS:"
echo "  Run API server:     python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo "  Run tests:          python -m pytest tests/"
echo "  Run Phase 2 tests:  python tests/phase2_validation.py"
echo "  Download historical data: python tests/data/historical/download_historical_tles.py"

echo ""
echo "🧪 TESTING MODE:"
echo "  The system runs in demo mode by default"
echo "  No external credentials required for basic testing"
echo "  Synthetic data used for demonstrations"

echo ""
echo "🔒 PRODUCTION DEPLOYMENT:"
echo "  1. Set real database credentials in .env"
echo "  2. Configure Space-Track API credentials"  
echo "  3. Set secure JWT_SECRET_KEY"
echo "  4. Review security settings"
echo "  5. Deploy with proper access controls"

echo ""
echo "✅ Deployment preparation complete!"