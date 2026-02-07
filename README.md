# SharpFlow

Polymarket Sharp Wallet Intelligence System.

Identifies, scores, and monitors the sharpest wallets on Polymarket using a hybrid scoring algorithm (CLV + Timing Alpha + Consistency + ROI).

## Deploy on Railway

1. Fork this repo
2. Connect to Railway
3. Add environment variables (optional):
   - `TELEGRAM_BOT_TOKEN` — for alerts
   - `TELEGRAM_CHAT_ID` — for alerts
4. Deploy

The app will automatically start fetching data from Polymarket and scoring wallets. First run takes ~10-15 minutes.

## API Endpoints

- `GET /` — Dashboard
- `GET /api/wallets` — Scored wallet leaderboard
- `GET /api/convergence` — Convergence signals
- `GET /api/stats` — System status
