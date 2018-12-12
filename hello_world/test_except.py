import time
import os

for idx in range(999):
    try:
        time.sleep(1)
        a=1/(idx%10)
        print(idx)
    except KeyboardInterrupt:
        print("manual cancel!")
        break
    except Exception as e:
        print(e)
    except:
        print("except!")
