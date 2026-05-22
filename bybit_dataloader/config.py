from pybit.unified_trading import HTTP
from datetime import datetime, timezone

session = HTTP(testnet=False)

symbols = [ "BTCUSDT", "SOLUSDT", "ETHUSDT", "XRPUSDT", "DOGEUSDT" ]

end_date = datetime.now()
start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

interval = "240"