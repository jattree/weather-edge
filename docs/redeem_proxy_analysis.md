# Proxy Wallet CTF Redemption: Technical Analysis

## The Problem

The current `executor.py` (line 867-869) creates a contract instance at the **proxy address** with the `redeemPositions` ABI and tries to call it directly:

```python
proxy_contract = w3.eth.contract(
    address=Web3.to_checksum_address(proxy),
    abi=REDEEM_ABI,
)
# ...
tx = proxy_contract.functions.redeemPositions(...).build_transaction({
    "from": account.address,  # EOA signs the tx
    ...
})
```

This fails because:
1. The proxy (`0xYOUR_PROXY_WALLET`) is an EIP-1167 minimal proxy pointing to implementation `0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f`
2. The implementation only has `initialize()`, `onERC1155Received`, and `onERC1155BatchReceived`, **no `execute()` function, no `redeemPositions()`**
3. The EOA (`0xYOUR_EOA_WALLET`) cannot sign transactions "as" the proxy, when the EOA sends a tx to the proxy address calling `redeemPositions`, the proxy delegates to its implementation which doesn't have that function
4. `CTF.redeemPositions` needs `msg.sender` to be the token holder (the proxy), but the EOA can't make the proxy call the CTF

## The Solution: Polymarket Relayer

The Polymarket Relayer is the **official mechanism** for executing transactions through proxy/Safe wallets without paying gas and without needing an `execute()` function on the proxy.

### How It Works

1. Your app creates a transaction (target contract + encoded calldata)
2. The user signs it with their private key
3. You submit it to Polymarket's relayer API
4. The relayer submits it on-chain, executing **from the proxy wallet** (msg.sender = proxy)
5. Polymarket pays the gas

### Authentication Options

**Option A: Builder API Keys** (for Builder Program members)
- Requires: `POLY_BUILDER_API_KEY`, `POLY_BUILDER_SECRET`, `POLY_BUILDER_PASSPHRASE`
- Uses HMAC-SHA256 signed headers
- Works with the `@polymarket/builder-relayer-client` SDK

**Option B: Relayer API Keys** (simpler, for anyone)
- Create from https://polymarket.com/settings?tab=api-keys
- Headers: `RELAYER_API_KEY`, `RELAYER_API_KEY_ADDRESS`
- Can use the SDK or call the REST API directly

### Contract Details

| Contract | Address |
|---|---|
| CTF (Conditional Tokens) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` |
| USDC.e (collateral) | `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174` |
| CTF Exchange | `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E` |
| Neg Risk CTF Exchange | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |
| Neg Risk Adapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` |
| Polymarket Proxy Factory | `0xaB45c5A4B0c941a2F231C04C3f49182e1A254052` |
| Relayer endpoint | `https://relayer-v2.polymarket.com/` |

### Redemption Function Signature

```solidity
function redeemPositions(
    address collateralToken,     // 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174 (USDC.e)
    bytes32 parentCollectionId,  // 0x0000...0000 (32 zero bytes)
    bytes32 conditionId,         // The market's condition ID
    uint256[] indexSets          // [1, 2] redeems both outcomes (only winning pays out)
)
```

- Called on the **CTF contract** (`0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`)
- Burns your entire token balance for the condition (no amount parameter)
- Winning tokens return $1.00 USDC.e each; losing tokens return $0

### TypeScript Implementation (from docs)

```typescript
import { encodeFunctionData } from "viem";

const CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045";
const USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174";

const redeemTx = {
  to: CTF_ADDRESS,
  data: encodeFunctionData({
    abi: [{
      name: "redeemPositions",
      type: "function",
      inputs: [
        { name: "collateralToken", type: "address" },
        { name: "parentCollectionId", type: "bytes32" },
        { name: "conditionId", type: "bytes32" },
        { name: "indexSets", type: "uint256[]" },
      ],
      outputs: [],
    }],
    functionName: "redeemPositions",
    args: [USDC, parentCollectionId, conditionId, indexSets],
  }),
  value: "0",
};

const response = await client.execute([redeemTx], "Redeem positions");
await response.wait();
```

### Python Implementation for executor.py

```python
# Instead of calling redeemPositions on the proxy contract directly,
# encode the calldata and submit via the Polymarket Relayer API.

from web3 import Web3
import httpx

CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
PARENT_COLLECTION = b"\x00" * 32
RELAYER_URL = "https://relayer-v2.polymarket.com/"

REDEEM_ABI = [{
    "name": "redeemPositions",
    "type": "function",
    "inputs": [
        {"name": "collateralToken", "type": "address"},
        {"name": "parentCollectionId", "type": "bytes32"},
        {"name": "conditionId", "type": "bytes32"},
        {"name": "indexSets", "type": "uint256[]"},
    ],
    "outputs": [],
}]

w3 = Web3()
ctf_contract = w3.eth.contract(
    address=Web3.to_checksum_address(CTF_ADDRESS),
    abi=REDEEM_ABI,
)

# Encode the calldata (target = CTF contract, not the proxy)
calldata = ctf_contract.functions.redeemPositions(
    Web3.to_checksum_address(USDC),
    PARENT_COLLECTION,
    cid_bytes,
    index_sets,
).build_transaction({"gas": 0})["data"]

# Submit via relayer
# The transaction structure for the relayer:
transaction = {
    "to": CTF_ADDRESS,
    "data": calldata,
    "value": "0",
}

# Using Relayer API Key auth:
headers = {
    "RELAYER_API_KEY": "<your-relayer-api-key>",
    "RELAYER_API_KEY_ADDRESS": "<your-eoa-address>",
}

# Using py-builder-relayer-client SDK:
# pip install polymarket-builder-relayer-client polymarket-builder-signing-sdk
# from polymarket_builder_relayer_client import RelayClient
# response = await client.execute([transaction], "Redeem positions")
# result = await response.wait()
```

### Relayer REST API (Direct)

```
POST https://relayer-v2.polymarket.com/
```

See the [Relayer API Reference](https://docs.polymarket.com/api-reference/relayer/submit-a-transaction) for exact request/response formats.

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/` | Submit a transaction |
| GET | `/{id}` | Get transaction by ID |
| GET | `/user/{address}` | Get recent transactions for user |
| GET | `/nonce/{address}` | Get current nonce |
| GET | `/relayer` | Get relayer address and nonce |

### Wallet Type Configuration

When initializing the CLOB client for a POLY_PROXY wallet:
- `signature_type = 1` (POLY_PROXY)
- `funder = "0xYOUR_PROXY_WALLET"` (the proxy address)

### Transaction States

| State | Terminal | Description |
|---|---|---|
| STATE_NEW | No | Received by relayer |
| STATE_EXECUTED | No | Submitted onchain |
| STATE_MINED | No | Included in a block |
| STATE_CONFIRMED | Yes | Finalized successfully |
| STATE_FAILED | Yes | Failed permanently |
| STATE_INVALID | Yes | Rejected as invalid |

## Implementation Plan

1. **Get Relayer API Key**: Create one at https://polymarket.com/settings?tab=api-keys OR use Builder API credentials if enrolled in Builder Program
2. **Install SDK**: `pip install polymarket-builder-relayer-client polymarket-builder-signing-sdk`
3. **Rewrite `redeem_positions()`**: Instead of building raw transactions signed by the EOA targeting the proxy address, encode `redeemPositions` calldata targeting the CTF contract and submit through the relayer
4. **The relayer executes from the proxy wallet**, making `msg.sender = proxy` which holds the tokens
5. **No gas needed**: Polymarket pays all gas fees
6. **Batch support**: Multiple redeem calls can be batched into a single `execute()` call

## Why the Current Approach Cannot Work

The proxy at `0xYOUR_PROXY_WALLET`:
- Is an EIP-1167 minimal clone of `0x44e999d5c2f66ef0861317f9a4805ac2e90aeb4f`
- The implementation has NO general-purpose `execute(address,bytes)` function
- It only has `initialize()` + ERC1155 receiver hooks
- You CANNOT call arbitrary functions through it by sending transactions to it
- The ONLY way to execute transactions "as" this proxy is through Polymarket's relayer infrastructure, which has the authority to execute calls from proxy wallets
