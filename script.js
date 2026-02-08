// Simulated Data
const ATTACKS = ["DDoS", "MITM", "Spoofing", "Injection", "Command Attack"];
const SEVERITY = ["Low", "Medium", "High", "Critical"];
const PUMPS = ["Pump A – ICU", "Pump B – Surgery", "Pump C – Pediatrics"];
const latencyEl = document.getElementById("latency");
// DOM Elements
const pumpBody = document.querySelector("#pumpTable tbody");
const alertBody = document.querySelector("#alertTable tbody");
const notification = document.getElementById("notification");

// Generate random pump status
function loadPumpStatus() {
    pumpBody.innerHTML = "";

    PUMPS.forEach(pump => {
        const hr = Math.floor(Math.random() * 100) + 40;
        const battery = Math.floor(Math.random() * 60) + 40;
        const flow = (Math.random() * 6 + 3).toFixed(1);
        const state = ["Normal", "Warning", "Critical"][Math.floor(Math.random()*3)];

        const row = `
            <tr>
                <td>${pump}</td>
                <td>${hr}</td>
                <td>${battery}%</td>
                <td>${flow}</td>
                <td>${state}</td>
            </tr>
        `;
        pumpBody.innerHTML += row;
    });
}

ws.onmessage = (msg) => {
  let e;
  try {
    e = JSON.parse(msg.data);
  } catch (err) {
    console.log("WS JSON parse error:", err, msg.data);
    return;
  }

  // ✅ DEBUG: check if server_ts_ms is present
  // (remove later)
  console.log("WS event:", e);

  // ✅ Latency for real prediction events only
  if (e.type === "event") {
    // server_ts_ms might arrive as number OR string → handle both
    const ts = (typeof e.server_ts_ms === "number")
      ? e.server_ts_ms
      : parseInt(e.server_ts_ms, 10);

    if (!Number.isNaN(ts)) {
      const latencyMs = Date.now() - ts;
      latencyEl.textContent = `Latency: ${latencyMs} ms`;
    } else if (e.server_ts_iso) {
      // fallback if only ISO exists
      const parsed = Date.parse(e.server_ts_iso);
      if (!Number.isNaN(parsed)) {
        const latencyMs = Date.now() - parsed;
        latencyEl.textContent = `Latency: ${latencyMs} ms`;
      }
    }
  }

  // keep your existing consume logic
  consume(e);
};


// Generate random IDS alerts
function loadIDSAlerts() {
    alertBody.innerHTML = "";

    // 35% chance of an alert
    if (Math.random() < 0.35) {
        const alert = {
            time: new Date().toLocaleTimeString(),
            attack: ATTACKS[Math.floor(Math.random() * ATTACKS.length)],
            severity: SEVERITY[Math.floor(Math.random() * SEVERITY.length)],
            pump: PUMPS[Math.floor(Math.random() * PUMPS.length)]
        };

        const row = `
            <tr>
                <td>${alert.time}</td>
                <td>${alert.attack}</td>
                <td>${alert.severity}</td>
                <td>${alert.pump}</td>
            </tr>
        `;
        alertBody.innerHTML += row;

        // Update doctor alert box
        notification.className = "notification red";
        notification.innerHTML = `⚠ ALERT: ${alert.attack} detected on ${alert.pump}!`;
    } 
    else {
        notification.className = "notification green";
        notification.innerHTML = "No security threats detected. All pumps are operating normally.";
    }
}

// Auto-refresh every 2 seconds
setInterval(() => {
    loadPumpStatus();
    loadIDSAlerts();
}, 2000);

// Initial load
loadPumpStatus();
loadIDSAlerts();

