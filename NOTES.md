# Development Notes

## TODOs

### High Priority
- [ ] Fix Neo4j connection issue on first startup (requires restart)
- [ ] Add batch processing endpoint for bulk claim analysis
- [ ] Improve error handling in CV engine when image quality is low
- [ ] Add rate limiting per API key instead of global limit

### Medium Priority
- [ ] Support PDF documents (currently only images)
- [ ] Add more languages for embeddings (Hindi, Tamil, Telugu)
- [ ] Improve graph visualizations in frontend
- [ ] Add model retraining pipeline
- [ ] Better caching strategy for document verification

### Low Priority
- [ ] Add dark mode to Streamlit UI
- [ ] Export analysis results to PDF
- [ ] Add email notifications for high-risk claims
- [ ] Improve documentation with more examples

## Known Issues

1. **Neo4j Connection**
   - Sometimes fails on first launch
   - Workaround: Restart the docker container
   - Need to investigate connection pooling

2. **Large PDFs**
   - Files >10MB timeout on verification
   - Consider streaming or chunking approach

3. **Hinglish Embeddings**
   - Works best with 50/50 mix of Hindi-English
   - Pure Hindi or pure English reduces accuracy
   - Might need to train custom model

4. **Generic Document Detector**
   - Accuracy varies significantly by document type
   - Hospital bills: ~85% accuracy
   - Driving licenses: ~78% accuracy
   - Need more training data

## Performance Notes

- ML inference is fast (~80ms) thanks to CatBoost
- Document verification can be slow with large images
- Redis caching helps a lot with repeated queries
- LLM explanations add 2-3s but worth it for UX

## Recent Changes

### Dec 2025
- Fixed feature alignment bug in ML engine
- Column name mismatch between dataset and model
- Added feature validation before prediction

### Nov 2025
- Integrated Groq LLM for explanations
- Much faster than OpenAI (10x speedup)
- Using Llama-3.3-70B model

### Oct 2025
- Added generic document detector
- Supports more document types now
- Still needs improvement

## Dev Environment Setup

Personal notes for development:

```bash
# My usual workflow
source venv/bin/activate
export GROQ_API_KEY=...
uvicorn api.main:app --reload &
streamlit run frontend/streamlit_app.py
```

# Debugging tips
- Set ENABLE_LLM_EXPLANATIONS=false to speed up testing
- Use /api/ml/score/detailed for feature importance
- Check Redis cache stats at /api/cache/stats

## Future Ideas

- Agentic architecture with LangGraph
- Real-time fraud pattern search via web
- Mobile app for field adjusters
- Integration with existing claims management systems
- Anomaly detection for unusual patterns
