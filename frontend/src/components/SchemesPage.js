import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useTranslation } from '../translations';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function SchemesPage({ user, onLogout }) {
  const navigate = useNavigate();
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const t = useTranslation(currentLanguage) || {};
  const [farmerDetails, setFarmerDetails] = useState({
    state: '',
    district: '',
    landSize: '',
    cropTypes: ''
  });

  // Indian states and their districts (all 28 states + 8 union territories)
  const stateDistrictMap = {
    'andhra-pradesh': ['Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna', 'Kurnool', 'Prakasam', 'Srikakulam', 'Visakhapatnam', 'Vizianagaram', 'West Godavari', 'YSR Kadapa'],
    'arunachal-pradesh': ['Anjaw', 'Changlang', 'Dibang Valley', 'East Kameng', 'East Siang', 'Kamle', 'Kra Daadi', 'Kurung Kumey', 'Lepa Rada', 'Lohit', 'Longding', 'Lower Dibang Valley', 'Lower Siang', 'Lower Subansiri', 'Namsai', 'Pakke Kessang', 'Papum Pare', 'Shi Yomi', 'Siang', 'Tawang', 'Tirap', 'Upper Siang', 'Upper Subansiri', 'West Kameng', 'West Siang'],
    'assam': ['Baksa', 'Barpeta', 'Biswanath', 'Bongaigaon', 'Cachar', 'Charaideo', 'Chirang', 'Darrang', 'Dhemaji', 'Dhubri', 'Dibrugarh', 'Goalpara', 'Golaghat', 'Hailakandi', 'Hojai', 'Jorhat', 'Kamrup', 'Kamrup Metropolitan', 'Karbi Anglong', 'Karimganj', 'Kokrajhar', 'Lakhimpur', 'Majuli', 'Morigaon', 'Nagaon', 'Nalbari', 'Dima Hasao', 'Sivasagar', 'Sonitpur', 'South Salmara-Mankachar', 'Tinsukia', 'Udalguri', 'West Karbi Anglong'],
    'bihar': ['Araria', 'Arwal', 'Aurangabad', 'Banka', 'Begusarai', 'Bhagalpur', 'Bhojpur', 'Buxar', 'Darbhanga', 'East Champaran', 'Gaya', 'Gopalganj', 'Jamui', 'Jehanabad', 'Kaimur', 'Katihar', 'Khagaria', 'Kishanganj', 'Lakhisarai', 'Madhepura', 'Madhubani', 'Munger', 'Muzaffarpur', 'Nalanda', 'Nawada', 'Patna', 'Purnia', 'Rohtas', 'Saharsa', 'Samastipur', 'Saran', 'Sheikhpura', 'Sheohar', 'Sitamarhi', 'Siwan', 'Supaul', 'Vaishali', 'West Champaran'],
    'chhattisgarh': ['Balod', 'Baloda Bazar', 'Balrampur', 'Bastar', 'Bemetara', 'Bijapur', 'Bilaspur', 'Dantewada', 'Dhamtari', 'Durg', 'Gariaband', 'Gaurela Pendra Marwahi', 'Janjgir Champa', 'Jashpur', 'Kabirdham', 'Kanker', 'Kondagaon', 'Korba', 'Koriya', 'Mahasamund', 'Mungeli', 'Narayanpur', 'Raigarh', 'Raipur', 'Rajnandgaon', 'Sukma', 'Surajpur', 'Surguja'],
    'goa': ['North Goa', 'South Goa'],
    'gujarat': ['Ahmedabad', 'Amreli', 'Anand', 'Aravalli', 'Banaskantha', 'Bharuch', 'Bhavnagar', 'Botad', 'Chhota Udepur', 'Dahod', 'Dang', 'Devbhoomi Dwarka', 'Gandhinagar', 'Gir Somnath', 'Jamnagar', 'Junagadh', 'Kachchh', 'Kheda', 'Mahisagar', 'Mehsana', 'Morbi', 'Narmada', 'Navsari', 'Panchmahal', 'Patan', 'Porbandar', 'Rajkot', 'Sabarkantha', 'Surat', 'Surendranagar', 'Tapi', 'Vadodara', 'Valsad'],
    'haryana': ['Ambala', 'Bhiwani', 'Charkhi Dadri', 'Faridabad', 'Fatehabad', 'Gurugram', 'Hisar', 'Jhajjar', 'Jind', 'Kaithal', 'Karnal', 'Kurukshetra', 'Mahendragarh', 'Nuh', 'Palwal', 'Panchkula', 'Panipat', 'Rewari', 'Rohtak', 'Sirsa', 'Sonipat', 'Yamunanagar'],
    'himachal-pradesh': ['Bilaspur', 'Chamba', 'Hamirpur', 'Kangra', 'Kinnaur', 'Kullu', 'Lahaul Spiti', 'Mandi', 'Shimla', 'Sirmaur', 'Solan', 'Una'],
    'jharkhand': ['Bokaro', 'Chatra', 'Deoghar', 'Dhanbad', 'Dumka', 'East Singhbhum', 'Garhwa', 'Giridih', 'Godda', 'Gumla', 'Hazaribagh', 'Jamtara', 'Khunti', 'Koderma', 'Latehar', 'Lohardaga', 'Pakur', 'Palamu', 'Ramgarh', 'Ranchi', 'Sahebganj', 'Seraikela Kharsawan', 'Simdega', 'West Singhbhum'],
    'karnataka': ['Bagalkot', 'Ballari', 'Belagavi', 'Bengaluru Rural', 'Bengaluru Urban', 'Bidar', 'Chamarajanagar', 'Chikballapur', 'Chikkamagaluru', 'Chitradurga', 'Dakshina Kannada', 'Davanagere', 'Dharwad', 'Gadag', 'Hassan', 'Haveri', 'Kalaburagi', 'Kodagu', 'Kolar', 'Koppal', 'Mandya', 'Mysuru', 'Raichur', 'Ramanagara', 'Shivamogga', 'Tumakuru', 'Udupi', 'Uttara Kannada', 'Vijayapura', 'Yadgir'],
    'kerala': ['Alappuzha', 'Ernakulam', 'Idukki', 'Kannur', 'Kasaragod', 'Kollam', 'Kottayam', 'Kozhikode', 'Malappuram', 'Palakkad', 'Pathanamthitta', 'Thiruvananthapuram', 'Thrissur', 'Wayanad'],
    'madhya-pradesh': ['Agar Malwa', 'Alirajpur', 'Anuppur', 'Ashoknagar', 'Balaghat', 'Barwani', 'Betul', 'Bhind', 'Bhopal', 'Burhanpur', 'Chhatarpur', 'Chhindwara', 'Damoh', 'Datia', 'Dewas', 'Dhar', 'Dindori', 'Guna', 'Gwalior', 'Harda', 'Hoshangabad', 'Indore', 'Jabalpur', 'Jhabua', 'Katni', 'Khandwa', 'Khargone', 'Mandla', 'Mandsaur', 'Morena', 'Narsinghpur', 'Neemuch', 'Panna', 'Raisen', 'Rajgarh', 'Ratlam', 'Rewa', 'Sagar', 'Satna', 'Sehore', 'Seoni', 'Shahdol', 'Shajapur', 'Sheopur', 'Shivpuri', 'Sidhi', 'Singrauli', 'Tikamgarh', 'Ujjain', 'Umaria', 'Vidisha'],
    'maharashtra': ['Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Mumbai City', 'Mumbai Suburban', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad', 'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim', 'Yavatmal'],
    'manipur': ['Bishnupur', 'Chandel', 'Churachandpur', 'Imphal East', 'Imphal West', 'Jiribam', 'Kakching', 'Kamjong', 'Kangpokpi', 'Noney', 'Pherzawl', 'Senapati', 'Tamenglong', 'Tengnoupal', 'Thoubal', 'Ukhrul'],
    'meghalaya': ['East Garo Hills', 'East Jaintia Hills', 'East Khasi Hills', 'North Garo Hills', 'Ri Bhoi', 'South Garo Hills', 'South West Garo Hills', 'South West Khasi Hills', 'West Garo Hills', 'West Jaintia Hills', 'West Khasi Hills'],
    'mizoram': ['Aizawl', 'Champhai', 'Hnahthial', 'Kolasib', 'Khawzawl', 'Lawngtlai', 'Lunglei', 'Mamit', 'Saiha', 'Saitual', 'Serchhip'],
    'nagaland': ['Dimapur', 'Kiphire', 'Kohima', 'Longleng', 'Mokokchung', 'Mon', 'Noklak', 'Peren', 'Phek', 'Tuensang', 'Wokha', 'Zunheboto'],
    'odisha': ['Angul', 'Balangir', 'Balasore', 'Bargarh', 'Bhadrak', 'Boudh', 'Cuttack', 'Deogarh', 'Dhenkanal', 'Gajapati', 'Ganjam', 'Jagatsinghpur', 'Jajpur', 'Jharsuguda', 'Kalahandi', 'Kandhamal', 'Kendrapara', 'Kendujhar', 'Khordha', 'Koraput', 'Malkangiri', 'Mayurbhanj', 'Nabarangpur', 'Nayagarh', 'Nuapada', 'Puri', 'Rayagada', 'Sambalpur', 'Subarnapur', 'Sundargarh'],
    'punjab': ['Amritsar', 'Barnala', 'Bathinda', 'Faridkot', 'Fatehgarh Sahib', 'Fazilka', 'Ferozepur', 'Gurdaspur', 'Hoshiarpur', 'Jalandhar', 'Kapurthala', 'Ludhiana', 'Mansa', 'Moga', 'Muktsar', 'Nawanshahr', 'Pathankot', 'Patiala', 'Rupnagar', 'Sangrur', 'Tarn Taran'],
    'rajasthan': ['Ajmer', 'Alwar', 'Banswara', 'Baran', 'Barmer', 'Bharatpur', 'Bhilwara', 'Bikaner', 'Bundi', 'Chittorgarh', 'Churu', 'Dausa', 'Dholpur', 'Dungarpur', 'Hanumangarh', 'Jaipur', 'Jaisalmer', 'Jalore', 'Jhalawar', 'Jhunjhunu', 'Jodhpur', 'Karauli', 'Kota', 'Nagaur', 'Pali', 'Pratapgarh', 'Rajsamand', 'Sawai Madhopur', 'Sikar', 'Sirohi', 'Sri Ganganagar', 'Tonk', 'Udaipur'],
    'sikkim': ['East Sikkim', 'North Sikkim', 'South Sikkim', 'West Sikkim'],
    'tamil-nadu': ['Ariyalur', 'Chengalpattu', 'Chennai', 'Coimbatore', 'Cuddalore', 'Dharmapuri', 'Dindigul', 'Erode', 'Kallakurichi', 'Kanchipuram', 'Kanyakumari', 'Karur', 'Krishnagiri', 'Madurai', 'Mayiladuthurai', 'Nagapattinam', 'Namakkal', 'Nilgiris', 'Perambalur', 'Pudukkottai', 'Ramanathapuram', 'Ranipet', 'Salem', 'Sivaganga', 'Tenkasi', 'Thanjavur', 'Theni', 'Thoothukudi', 'Tiruchirappalli', 'Tirunelveli', 'Tirupathur', 'Tiruppur', 'Tiruvallur', 'Tiruvannamalai', 'Tiruvarur', 'Vellore', 'Viluppuram', 'Virudhunagar'],
    'telangana': ['Adilabad', 'Bhadradri Kothagudem', 'Hyderabad', 'Jagtial', 'Jangaon', 'Jayashankar', 'Jogulamba', 'Kamareddy', 'Karimnagar', 'Khammam', 'Komaram Bheem', 'Mahabubabad', 'Mahbubnagar', 'Mancherial', 'Medak', 'Medchal', 'Mulugu', 'Nagarkurnool', 'Nalgonda', 'Narayanpet', 'Nirmal', 'Nizamabad', 'Peddapalli', 'Rajanna Sircilla', 'Ranga Reddy', 'Sangareddy', 'Siddipet', 'Suryapet', 'Vikarabad', 'Wanaparthy', 'Warangal Rural', 'Warangal Urban', 'Yadadri Bhuvanagiri'],
    'tripura': ['Dhalai', 'Gomati', 'Khowai', 'North Tripura', 'Sepahijala', 'South Tripura', 'Unakoti', 'West Tripura'],
    'uttar-pradesh': ['Agra', 'Aligarh', 'Ambedkar Nagar', 'Amethi', 'Amroha', 'Auraiya', 'Ayodhya', 'Azamgarh', 'Baghpat', 'Bahraich', 'Ballia', 'Balrampur', 'Banda', 'Barabanki', 'Bareilly', 'Basti', 'Bhadohi', 'Bijnor', 'Budaun', 'Bulandshahr', 'Chandauli', 'Chitrakoot', 'Deoria', 'Etah', 'Etawah', 'Farrukhabad', 'Fatehpur', 'Firozabad', 'Gautam Buddha Nagar', 'Ghaziabad', 'Ghazipur', 'Gonda', 'Gorakhpur', 'Hamirpur', 'Hapur', 'Hardoi', 'Hathras', 'Jalaun', 'Jaunpur', 'Jhansi', 'Kannauj', 'Kanpur Dehat', 'Kanpur Nagar', 'Kasganj', 'Kaushambi', 'Kheri', 'Kushinagar', 'Lalitpur', 'Lucknow', 'Maharajganj', 'Mahoba', 'Mainpuri', 'Mathura', 'Mau', 'Meerut', 'Mirzapur', 'Moradabad', 'Muzaffarnagar', 'Pilibhit', 'Pratapgarh', 'Prayagraj', 'Raebareli', 'Rampur', 'Saharanpur', 'Sambhal', 'Sant Kabir Nagar', 'Shahjahanpur', 'Shamli', 'Shrawasti', 'Siddharthnagar', 'Sitapur', 'Sonbhadra', 'Sultanpur', 'Unnao', 'Varanasi'],
    'uttarakhand': ['Almora', 'Bageshwar', 'Chamoli', 'Champawat', 'Dehradun', 'Haridwar', 'Nainital', 'Pauri', 'Pithoragarh', 'Rudraprayag', 'Tehri', 'Udham Singh Nagar', 'Uttarkashi'],
    'west-bengal': ['Alipurduar', 'Bankura', 'Birbhum', 'Cooch Behar', 'Dakshin Dinajpur', 'Darjeeling', 'Hooghly', 'Howrah', 'Jalpaiguri', 'Jhargram', 'Kalimpong', 'Kolkata', 'Malda', 'Murshidabad', 'Nadia', 'North 24 Parganas', 'Paschim Bardhaman', 'Paschim Medinipur', 'Purba Bardhaman', 'Purba Medinipur', 'Purulia', 'South 24 Parganas', 'Uttar Dinajpur'],
    // Union Territories
    'andaman-nicobar': ['Nicobar', 'North Middle Andaman', 'South Andaman'],
    'chandigarh': ['Chandigarh'],
    'dadra-nagar-haveli-daman-diu': ['Dadra Nagar Haveli', 'Daman', 'Diu'],
    'delhi': ['Central Delhi', 'East Delhi', 'New Delhi', 'North Delhi', 'North East Delhi', 'North West Delhi', 'Shahdara', 'South Delhi', 'South East Delhi', 'South West Delhi', 'West Delhi'],
    'jammu-kashmir': ['Anantnag', 'Bandipora', 'Baramulla', 'Budgam', 'Doda', 'Ganderbal', 'Jammu', 'Kathua', 'Kishtwar', 'Kulgam', 'Kupwara', 'Poonch', 'Pulwama', 'Rajouri', 'Ramban', 'Reasi', 'Samba', 'Shopian', 'Srinagar', 'Udhampur'],
    'ladakh': ['Kargil', 'Leh'],
    'lakshadweep': ['Lakshadweep'],
    'puducherry': ['Karaikal', 'Mahe', 'Puducherry', 'Yanam']
  };
  const [schemes, setSchemes] = useState([]);
  const [appliedSchemes, setAppliedSchemes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showForm, setShowForm] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('available'); // 'available' or 'applied'

  useEffect(() => {
    const savedLanguage = localStorage.getItem('uiLanguage') || 'en';
    setCurrentLanguage(savedLanguage);
  }, []);

  const handleBackToChat = () => {
    navigate('/');
  };

  const handleApplyScheme = (scheme) => {
    // Move scheme from available to applied
    const appliedScheme = {
      ...scheme,
      enrollment_status: {
        status: 'pending',
        application_date: new Date().toISOString(),
        application_id: `APP${Math.random().toString(36).substr(2, 9).toUpperCase()}`
      }
    };

    // Add to applied schemes
    setAppliedSchemes(prev => [...prev, appliedScheme]);

    // Remove from available schemes
    setSchemes(prev => prev.filter(s => s.id !== scheme.id));

    // Show success message
    alert(`Successfully applied for ${scheme.name}! Your application is now pending review.`);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');

      // Convert cropTypes string to array
      const formData = {
        ...farmerDetails,
        landSize: parseFloat(farmerDetails.landSize),
        cropTypes: [farmerDetails.cropTypes] // Convert single selection to array
      };

      const response = await axios.post(`${API}/schemes/find`, formData, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (response.data.success) {
        setSchemes(response.data.schemes);
        setShowForm(false);
      } else {
        setError('Failed to find schemes. Please try again.');
      }

    } catch (err) {
      console.error('Failed to find schemes:', err);
      setError('Failed to connect to server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFarmerDetails(prev => ({
      ...prev,
      [name]: value,
      // Reset district when state changes
      ...(name === 'state' && { district: '' })
    }));
  };

  return (
    <div className="schemes-page">
      <div className="page-header">
        <button className="back-button" onClick={handleBackToChat}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          {t.backToChat || 'Back to Chat'}
        </button>
        <h1>🌾 {t.schemesTitle || 'Government Schemes'}</h1>
        <p>Discover agricultural schemes and subsidies you're eligible for</p>
      </div>

      <div className="schemes-content">
        {error && (
          <div className="error-message">
            <p>{error}</p>
            <button onClick={() => setError(null)}>{t.dismiss || 'Dismiss'}</button>
          </div>
        )}

        {showForm ? (
          <div className="farmer-details-form">
            <h2>{t.tellUsAboutFarm || 'Tell us about your farm'}</h2>
            <form onSubmit={handleFormSubmit}>
              <div className="form-group">
                <label htmlFor="state">{t.state || 'State'}</label>
                <select
                  id="state"
                  name="state"
                  value={farmerDetails.state}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">{t.selectState || 'Select State'}</option>
                  {/* States */}
                  <option value="andhra-pradesh">{t.stateAndhraPradesh || 'Andhra Pradesh'}</option>
                  <option value="arunachal-pradesh">{t.stateArunachalPradesh || 'Arunachal Pradesh'}</option>
                  <option value="assam">{t.stateAssam || 'Assam'}</option>
                  <option value="bihar">{t.stateBihar || 'Bihar'}</option>
                  <option value="chhattisgarh">{t.stateChhattisgarh || 'Chhattisgarh'}</option>
                  <option value="goa">{t.stateGoa || 'Goa'}</option>
                  <option value="gujarat">{t.stateGujarat || 'Gujarat'}</option>
                  <option value="haryana">{t.stateHaryana || 'Haryana'}</option>
                  <option value="himachal-pradesh">{t.stateHimachalPradesh || 'Himachal Pradesh'}</option>
                  <option value="jharkhand">{t.stateJharkhand || 'Jharkhand'}</option>
                  <option value="karnataka">{t.stateKarnataka || 'Karnataka'}</option>
                  <option value="kerala">{t.stateKerala || 'Kerala'}</option>
                  <option value="madhya-pradesh">{t.stateMadhyaPradesh || 'Madhya Pradesh'}</option>
                  <option value="maharashtra">{t.stateMaharashtra || 'Maharashtra'}</option>
                  <option value="manipur">{t.stateManipur || 'Manipur'}</option>
                  <option value="meghalaya">{t.stateMeghalaya || 'Meghalaya'}</option>
                  <option value="mizoram">{t.stateMizoram || 'Mizoram'}</option>
                  <option value="nagaland">{t.stateNagaland || 'Nagaland'}</option>
                  <option value="odisha">{t.stateOdisha || 'Odisha'}</option>
                  <option value="punjab">{t.statePunjab || 'Punjab'}</option>
                  <option value="rajasthan">{t.stateRajasthan || 'Rajasthan'}</option>
                  <option value="sikkim">{t.stateSikkim || 'Sikkim'}</option>
                  <option value="tamil-nadu">{t.stateTamilNadu || 'Tamil Nadu'}</option>
                  <option value="telangana">{t.stateTelangana || 'Telangana'}</option>
                  <option value="tripura">{t.stateTripura || 'Tripura'}</option>
                  <option value="uttar-pradesh">{t.stateUttarPradesh || 'Uttar Pradesh'}</option>
                  <option value="uttarakhand">{t.stateUttarakhand || 'Uttarakhand'}</option>
                  <option value="west-bengal">{t.stateWestBengal || 'West Bengal'}</option>
                  {/* Union Territories */}
                  <option value="andaman-nicobar">{t.utAndamanNicobar || 'Andaman & Nicobar Islands'}</option>
                  <option value="chandigarh">{t.utChandigarh || 'Chandigarh'}</option>
                  <option value="dadra-nagar-haveli-daman-diu">{t.utDadraNagarHaveliDamanDiu || 'Dadra & Nagar Haveli and Daman & Diu'}</option>
                  <option value="delhi">{t.utDelhi || 'Delhi'}</option>
                  <option value="jammu-kashmir">{t.utJammuKashmir || 'Jammu & Kashmir'}</option>
                  <option value="ladakh">{t.utLadakh || 'Ladakh'}</option>
                  <option value="lakshadweep">{t.utLakshadweep || 'Lakshadweep'}</option>
                  <option value="puducherry">{t.utPuducherry || 'Puducherry'}</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="district">{t.district || 'District'}</label>
                <select
                  id="district"
                  name="district"
                  value={farmerDetails.district}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">{t.selectDistrict || 'Select District'}</option>
                  {farmerDetails.state && stateDistrictMap[farmerDetails.state]?.map(district => {
                    const districtKey = `district${district.replace(/\s+/g, '')}`;
                    return (
                      <option key={district} value={district.toLowerCase().replace(/\s+/g, '-')}>
                        {t[districtKey] || district}
                      </option>
                    );
                  })}
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="landSize">{t.landSize || 'Land Size (acres)'}</label>
                <input
                  type="number"
                  id="landSize"
                  name="landSize"
                  value={farmerDetails.landSize}
                  onChange={handleInputChange}
                  placeholder={t.landSizePlaceholder || 'Enter land size in acres'}
                  min="0.1"
                  step="0.1"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="cropTypes">{t.primaryCrops || 'Primary Crops'}</label>
                <select
                  id="cropTypes"
                  name="cropTypes"
                  value={farmerDetails.cropTypes}
                  onChange={handleInputChange}
                  required
                >
                  <option value="">{t.selectPrimaryCrop || 'Select Primary Crop'}</option>
                  <option value="wheat">{t.wheat || 'Wheat'}</option>
                  <option value="rice">{t.rice || 'Rice'}</option>
                  <option value="cotton">{t.cotton || 'Cotton'}</option>
                  <option value="sugarcane">{t.sugarcane || 'Sugarcane'}</option>
                  <option value="maize">{t.maize || 'Maize'}</option>
                </select>
              </div>

              <button type="submit" className="submit-button" disabled={loading}>
                {loading ? (t.findingSchemes || 'Finding Schemes...') : (t.findMySchemes || 'Find My Schemes')}
              </button>
            </form>
          </div>
        ) : (
          <div className="schemes-results">
            <div className="results-header">
              <h2>Government Schemes Finder</h2>
              <button className="edit-details-button" onClick={() => setShowForm(true)}>
                {t.editDetails || 'Edit Details'}
              </button>
            </div>

            {/* Tab Navigation */}
            <div className="tab-navigation">
              <button
                className={`tab-button ${activeTab === 'available' ? 'active' : ''}`}
                onClick={() => setActiveTab('available')}
              >
                {t.availableSchemes || 'Available Schemes'} ({schemes.length})
              </button>
              <button
                className={`tab-button ${activeTab === 'applied' ? 'active' : ''}`}
                onClick={() => setActiveTab('applied')}
              >
                {t.appliedSchemes || 'Applied Schemes'} ({appliedSchemes.length})
              </button>
            </div>

            {/* Available Schemes Tab */}
            {activeTab === 'available' ? (
              <div className="available-schemes-section">
                {schemes.length === 0 ? (
                  <div className="empty-schemes">
                    <p>No schemes found for your criteria. Try adjusting your farm details.</p>
                  </div>
                ) : (
                  <div className="schemes-grid">
                    {schemes.map(scheme => (
                      <div key={scheme.id} className="scheme-card eligible">
                        <div className="scheme-header">
                          <h3>{t[`scheme_${scheme.id}_name`] || scheme.name}</h3>
                          <span className="status-badge eligible">
                            📋 {t.eligible || 'Eligible'}
                          </span>
                        </div>

                        <p className="scheme-description">{t[`scheme_${scheme.id}_description`] || scheme.description}</p>

                        <div className="scheme-benefit">
                          <strong>{t.benefit}: {t[`scheme_${scheme.id}_benefit`] || scheme.benefit_description}</strong>
                        </div>

                        <div className="scheme-eligibility">
                          <h4>{t.eligibility || 'Eligibility'}:</h4>
                          <ul>
                            {scheme.eligibility_criteria.slice(0, 3).map((criteria, index) => (
                              <li key={index}>{t[`scheme_${scheme.id}_eligibility_${index}`] || criteria}</li>
                            ))}
                          </ul>
                        </div>

                        <div className="scheme-actions">
                          <button
                            className="scheme-action-button apply-button"
                            onClick={() => handleApplyScheme(scheme)}
                          >
                            {t.applyNow || 'Apply Now'}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              /* Applied Schemes Tab */
              <div className="applied-schemes-section">
                {appliedSchemes.length === 0 ? (
                  <div className="empty-schemes">
                    <p>You haven't applied to any schemes yet. Browse available schemes and apply!</p>
                  </div>
                ) : (
                  <div className="schemes-grid">
                    {appliedSchemes.map(scheme => (
                      <div key={scheme.id} className="scheme-card pending">
                        <div className="scheme-header">
                          <h3>{t[`scheme_${scheme.id}_name`] || scheme.name}</h3>
                          <span className="status-badge pending">
                            ⏳ {t.pending || 'Pending'}
                          </span>
                        </div>

                        <p className="scheme-description">{t[`scheme_${scheme.id}_description`] || scheme.description}</p>

                        <div className="scheme-benefit">
                          <strong>{t.benefit}: {t[`scheme_${scheme.id}_benefit`] || scheme.benefit_description}</strong>
                        </div>

                        <div className="application-info">
                          <small>{t.appliedOn || 'Applied on'}: {new Date(scheme.enrollment_status.application_date).toLocaleDateString()}</small>
                          <small>{t.applicationId || 'Application ID'}: {scheme.enrollment_status.application_id}</small>
                        </div>

                        <div className="scheme-actions">
                          <button
                            className="scheme-action-button status-button"
                            onClick={() => alert(`${t.applicationUnderReview || 'Application'} ${scheme.enrollment_status.application_id} ${t.isUnderReview || 'is under review. You will be notified once processed.'}`)}
                          >
                            {t.checkStatus || 'Check Status'}
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="benefits-summary">
              <h3>Your Benefits Summary</h3>
              <div className="summary-stats">
                <div className="stat">
                  <span className="stat-value">₹{schemes.reduce((total, scheme) => total + (scheme.enrollment_status?.amount_received || 0), 0).toFixed(0)}</span>
                  <span className="stat-label">Total Received This Year</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{schemes.filter(s => s.enrollment_status?.status === 'enrolled').length}</span>
                  <span className="stat-label">Enrolled Schemes</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{schemes.length}</span>
                  <span className="stat-label">Available Schemes</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SchemesPage;