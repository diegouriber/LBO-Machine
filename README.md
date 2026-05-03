# LBO Machine

LBO Machine is an NLP-based screening model for identifying potential leveraged buyout candidates from the S&P 500.

The project combines two layers:

1. **NLP stagnation layer**  
   Uses company filing-derived metrics to detect signs of strategic stagnation, such as declining innovation language, strategic decay, topic rigidity, and sentiment-growth misalignment.

2. **Financial feasibility layer**  
   Uses SEC financial statement data to evaluate whether a company has characteristics that could make it more suitable for LBO review, such as free cash flow strength, deleveraging capacity, manageable leverage, interest coverage, and low capital expenditure burden.

The final output is a ranked list of companies that show both:

- signs of strategic stagnation in their language, and
- enough financial feasibility to be worth deeper LBO analysis.

This model is not meant to make final investment decisions. It is a first-stage screening tool.

---

## Project Pipeline

The full pipeline follows this structure:

```text
S&P 500 universe
        ↓
SEC financial data extraction
        ↓
Financial feasibility scoring
        ↓
NLP stagnation metric cleaning
        ↓
NLP stagnation scoring
        ↓
Merge NLP and financial layers
        ↓
Final LBO candidate ranking
        ↓
Charts, summary tables, and validation outputs