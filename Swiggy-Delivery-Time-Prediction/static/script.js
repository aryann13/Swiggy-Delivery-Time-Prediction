/**
 * Swiggy Delivery Time Prediction — Frontend Logic
 * ==================================================
 * Handles form submission, payload construction, and result display.
 */

// ── DOM References ──
const form         = document.getElementById('predict-form');
const submitBtn    = document.getElementById('btn-submit');
const btnText      = document.getElementById('btn-text');
const resultBox    = document.getElementById('result-container');

// ── Auto-generate Hidden Fields ──
function generateID() {
    // Matches the raw dataset format: e.g. "5632A0"
    const hex = Math.floor(Math.random() * 0xFFFFFF).toString(16).toUpperCase().padStart(6, '0');
    return hex;
}

function generateRiderID(city) {
    // Matches format like "INDORES13DEL02"
    const cityMap = {
        'metropolitian': 'MUMBA',
        'urban':         'INDOR',
        'semi-urban':    'COIM'
    };
    const prefix = cityMap[city] || 'METRO';
    const num = String(Math.floor(Math.random() * 20) + 1).padStart(2, '0');
    return `${prefix}RES${num}DEL${num}`;
}


// ── Build JSON Payload from Form ──
function buildPayload() {
    const get = (id) => document.getElementById(id).value;

    const city = get('city');

    // Format the date from yyyy-mm-dd (HTML input) → dd-mm-yyyy (raw data format)
    const rawDate = get('order_date');        // "2022-03-19"
    const [y, m, d] = rawDate.split('-');
    const formattedDate = `${d}-${m}-${y}`;  // "19-03-2022"

    // Time fields — raw data uses "HH:MM" strings
    const orderTime  = get('order_time');       // "11:00"
    const pickupTime = get('pickup_time');      // "11:15"

    return {
        ID:                          generateID(),
        Delivery_person_ID:          generateRiderID(city),
        Delivery_person_Age:         get('age'),
        Delivery_person_Ratings:     get('ratings'),
        Restaurant_latitude:         parseFloat(get('rest_lat')),
        Restaurant_longitude:        parseFloat(get('rest_lng')),
        Delivery_location_latitude:  parseFloat(get('del_lat')),
        Delivery_location_longitude: parseFloat(get('del_lng')),
        Order_Date:                  formattedDate,
        Time_Orderd:                 orderTime,
        Time_Order_picked:           pickupTime,
        Weatherconditions:           `conditions ${get('weather')}`,
        Road_traffic_density:        get('traffic'),
        Vehicle_condition:           parseInt(get('vehicle_condition')),
        Type_of_order:               get('type_of_order'),
        Type_of_vehicle:             get('type_of_vehicle'),
        multiple_deliveries:         get('multiple_deliveries'),
        Festival:                    get('festival'),
        City:                        get('city')
    };
}


// ── Display Result ──
function showResult(minutes) {
    resultBox.innerHTML = `
        <div class="result-card">
            <div class="result-card__label">Estimated Delivery Time</div>
            <div class="result-card__value">
                ${minutes}<span class="result-card__unit">min</span>
            </div>
            <div class="result-card__desc">Predicted by Stacking Regressor (RF + XGBoost)</div>
        </div>
    `;
}

function showError(message) {
    resultBox.innerHTML = `
        <div class="error-card">
            <div class="error-card__msg">⚠ ${message}</div>
        </div>
    `;
}

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
        btnText.innerHTML = '<span class="spinner"></span> Predicting…';
    } else {
        btnText.textContent = 'Predict Delivery Time';
    }
}


// ── Form Submission ──
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);
    resultBox.innerHTML = '';

    try {
        const payload = buildPayload();
        console.log('Sending payload:', payload);

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const prediction = await response.json();
        showResult(prediction);

    } catch (err) {
        console.error('Prediction failed:', err);
        showError(err.message || 'Something went wrong. Please try again.');
    } finally {
        setLoading(false);
    }
});


// ── Auto-fill Pickup Time (order_time + 10-15 min) ──
document.getElementById('order_time').addEventListener('change', (e) => {
    const pickupInput = document.getElementById('pickup_time');
    if (!pickupInput.value) {
        const [h, m] = e.target.value.split(':').map(Number);
        const offset = 10 + Math.floor(Math.random() * 6); // 10–15 min
        const totalMin = h * 60 + m + offset;
        const newH = String(Math.floor(totalMin / 60) % 24).padStart(2, '0');
        const newM = String(totalMin % 60).padStart(2, '0');
        pickupInput.value = `${newH}:${newM}`;
    }
});
