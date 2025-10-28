# Translation Guide for Schemes and Districts

## Overview

The schemes page and district dropdowns now support full translations. This guide explains how to add more translations.

## How It Works

### District Translations

Districts are translated using a naming convention:
```javascript
districtName: "Translated Name"
```

**Example:**
```javascript
// English
districtAmbala: "Ambala",
districtBhiwani: "Bhiwani",

// Hindi
districtAmbala: "अंबाला",
districtBhiwani: "भिवानी",
```

### Scheme Translations

Schemes use a structured translation key format:
```javascript
scheme_<scheme_id>_name: "Scheme Name"
scheme_<scheme_id>_description: "Description"
scheme_<scheme_id>_benefit: "Benefit details"
scheme_<scheme_id>_eligibility_0: "First eligibility criterion"
scheme_<scheme_id>_eligibility_1: "Second eligibility criterion"
scheme_<scheme_id>_eligibility_2: "Third eligibility criterion"
```

**Example: PM-KISAN Scheme**

English:
```javascript
scheme_pm_kisan_name: "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
scheme_pm_kisan_description: "Direct income support scheme for small and marginal farmers",
scheme_pm_kisan_benefit: "₹6,000 per year in 3 equal installments of ₹2,000 each",
scheme_pm_kisan_eligibility_0: "All landholding farmers (small and marginal)",
scheme_pm_kisan_eligibility_1: "Land records should be in farmer's name",
scheme_pm_kisan_eligibility_2: "Applicable across all states and UTs",
```

Hindi:
```javascript
scheme_pm_kisan_name: "पीएम-किसान (प्रधानमंत्री किसान सम्मान निधि)",
scheme_pm_kisan_description: "छोटे और सीमांत किसानों के लिए प्रत्यक्ष आय सहायता योजना",
scheme_pm_kisan_benefit: "₹6,000 प्रति वर्ष 3 समान किस्तों में ₹2,000 प्रत्येक",
scheme_pm_kisan_eligibility_0: "सभी भूमिधारक किसान (छोटे और सीमांत)",
scheme_pm_kisan_eligibility_1: "भूमि रिकॉर्ड किसान के नाम पर होना चाहिए",
scheme_pm_kisan_eligibility_2: "सभी राज्यों और केंद्र शासित प्रदेशों में लागू",
```

## Adding New Translations

### Step 1: Find the Scheme ID

The scheme ID is usually in the backend database. Common scheme IDs:
- `pm_kisan` - PM-KISAN scheme
- `pmfby` - Pradhan Mantri Fasal Bima Yojana
- `soil_health_card` - Soil Health Card scheme

### Step 2: Add to translations.js

Open `/frontend/src/translations.js` and add translations in both `en` and your target language sections.

**For English (en section):**
```javascript
// PM-FASAL-BIMA-YOJANA Scheme
scheme_pmfby_name: "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
scheme_pmfby_description: "Crop insurance scheme to protect farmers against crop loss",
scheme_pmfby_benefit: "Insurance coverage for pre-sowing to post-harvest risks",
scheme_pmfby_eligibility_0: "All farmers growing notified crops",
scheme_pmfby_eligibility_1: "Premium: 2% for Kharif, 1.5% for Rabi crops",
scheme_pmfby_eligibility_2: "Covers natural calamities, pests & diseases",
```

**For Hindi (hi section):**
```javascript
// PM-FASAL-BIMA-YOJANA Scheme
scheme_pmfby_name: "प्रधानमंत्री फसल बीमा योजना (PMFBY)",
scheme_pmfby_description: "फसल नुकसान से किसानों की सुरक्षा के लिए फसल बीमा योजना",
scheme_pmfby_benefit: "बुवाई से लेकर कटाई के बाद के जोखिमों के लिए बीमा कवरेज",
scheme_pmfby_eligibility_0: "अधिसूचित फसलें उगाने वाले सभी किसान",
scheme_pmfby_eligibility_1: "प्रीमियम: खरीफ के लिए 2%, रबी फसलों के लिए 1.5%",
scheme_pmfby_eligibility_2: "प्राकृतिक आपदाओं, कीटों और रोगों को कवर करता है",
```

### Step 3: Adding District Translations

To add district translations for a state:

1. Find the district name in `SchemesPage.js` in the `stateDistrictMap`
2. Convert to camelCase with "district" prefix
3. Add to translations.js

**Example for Punjab districts:**

```javascript
// English
districtAmritsar: "Amritsar",
districtLudhiana: "Ludhiana",
districtJalandhar: "Jalandhar",
districtPatiala: "Patiala",

// Hindi  
districtAmritsar: "अमृतसर",
districtLudhiana: "लुधियाना",
districtJalandhar: "जालंधर",
districtPatiala: "पटियाला",

// Punjabi (pa)
districtAmritsar: "ਅੰਮ੍ਰਿਤਸਰ",
districtLudhiana: "ਲੁਧਿਆਣਾ",
districtJalandhar: "ਜਲੰਧਰ",
districtPatiala: "ਪਟਿਆਲਾ",
```

## Current Translation Status

### ✅ Fully Translated
- **Haryana districts** (all 22 districts) - English & Hindi
- **PM-KISAN scheme** - English & Hindi
- **Common UI elements** - All 10 languages

### ⚠️ Partially Translated  
- Other state districts - Need translations
- Other government schemes - Need translations

### 📋 Fallback Behavior

If a translation is missing:
- The system automatically falls back to English
- No errors are shown
- Translation can be added later without breaking functionality

## Translation Keys Reference

### Common Scheme Keys
```javascript
eligibility: "Eligibility / पात्रता"
applyNow: "Apply Now / अभी आवेदन करें"
checkStatus: "Check Status / स्थिति जांचें"
appliedOn: "Applied on / आवेदन किया"
applicationId: "Application ID / आवेदन आईडी"
```

### States (Already Translated)
All 28 states + 8 UTs are translated in all 10 languages:
- English, Hindi, Tamil, Telugu, Marathi, Bengali, Gujarati, Kannada, Malayalam, Punjabi

## Tips for Translators

1. **Keep it concise**: Scheme names and benefits should be clear and brief
2. **Use official translations**: For government scheme names, use official translated names
3. **Maintain formatting**: Keep ₹ symbols and numbers as-is
4. **Test thoroughly**: Switch languages in the UI to verify translations

## Supported Languages

1. English (en)
2. Hindi (hi)
3. Tamil (ta)
4. Telugu (te)
5. Marathi (mr)
6. Bengali (bn)
7. Gujarati (gu)
8. Kannada (kn)
9. Malayalam (ml)
10. Punjabi (pa)

## Contributing Translations

To contribute translations:

1. Identify missing translations (check which schemes/districts show in English when another language is selected)
2. Add translations following the naming conventions above
3. Test by switching languages in the UI
4. Ensure consistency with existing translations

## Example: Adding a Complete Scheme Translation

```javascript
// In translations.js under 'en' object:
scheme_kcc_name: "Kisan Credit Card (KCC)",
scheme_kcc_description: "Provides credit to farmers for agricultural needs",
scheme_kcc_benefit: "Short-term credit at 7% interest with 3% subvention",
scheme_kcc_eligibility_0: "All farmers - owner cultivators and tenant farmers",
scheme_kcc_eligibility_1: "SHGs or Joint Liability Groups of farmers",
scheme_kcc_eligibility_2: "Valid for 5 years with annual review",

// In translations.js under 'hi' object:
scheme_kcc_name: "किसान क्रेडिट कार्ड (KCC)",
scheme_kcc_description: "किसानों को कृषि आवश्यकताओं के लिए ऋण प्रदान करता है",
scheme_kcc_benefit: "3% सब्वेंशन के साथ 7% ब्याज पर अल्पकालिक ऋण",
scheme_kcc_eligibility_0: "सभी किसान - मालिक खेतिहर और किरायेदार किसान",
scheme_kcc_eligibility_1: "किसानों के SHG या संयुक्त देयता समूह",
scheme_kcc_eligibility_2: "वार्षिक समीक्षा के साथ 5 वर्षों के लिए वैध",
```

## Need Help?

If you need help with translations:
1. Check existing translations in `translations.js` for patterns
2. Use official government scheme documentation for accurate translations
3. Test your translations in the UI before finalizing
