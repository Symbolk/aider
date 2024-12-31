# mall4cloud-payment

## 项目总结
项目 mall4cloud-payment 包含以下主要模块：
- 处理订单支付记录的业务逻辑，包括支付、支付成功处理及支付状态查询。
- 该模块负责处理支付请求、管理支付状态及定义支付相关的业务数据。
- 该模块用于定义和管理订单支付记录的相关数据库操作。
- 处理支付异步回调通知并更新支付状态，定义支付后返回的基础数据对象。
- 定义订单支付记录的数据传输对象，包含订单号和支付回跳地址。

## 项目概览

[点击在浏览器中打开](repo_overview_with_communities.html)

## 主要模块
### 支付管理
处理订单支付记录的业务逻辑，包括支付、支付成功处理及支付状态查询。

```mermaid
graph TD
    A[开始支付流程] --> B[创建支付记录]
    B --> C[发起支付请求]
    C --> D{支付成功?}
    D -- 是 --> E[更新支付状态为成功]
    E --> F[处理支付成功回调]
    F --> G[完成支付流程]
    D -- 否 --> H[更新支付状态为失败]
    H --> I[处理支付失败回调]
    I --> J[结束支付流程]

    style A fill:#f9f,stroke:#333,stroke-width:4px
    style G fill:#bbf,stroke:#333,stroke-width:4px
    style J fill:#bbf,stroke:#333,stroke-width:4px
```

```mermaid
graph TD
    A[用户发起支付] --> B[创建支付记录]
    B --> C[处理支付请求]
    C --> D{支付成功?}
    D -- 是 --> E[更新支付状态为成功]
    D -- 否 --> F[更新支付状态为失败]
    E --> G[通知用户支付成功]
    F --> H[通知用户支付失败]
    G --> I[查询支付状态]
    H --> I
    I --> J[返回支付状态信息]
```

### 支付处理
该模块负责处理支付请求、管理支付状态及定义支付相关的业务数据。

```mermaid
graph TD
    A[PayController] -->|接收支付请求| B[PayInfoBO]
    B -->|生成支付信息| C[支付接口]
    C -->|处理支付| D[支付状态查询]
    D -->|返回支付状态| A
```

### 支付记录管理
该模块用于定义和管理订单支付记录的相关数据库操作。

```mermaid
graph TD
    A[PayInfoService] -->|调用| B[PayInfoMapper]
    B -->|执行数据库操作| C[数据库]
    A -->|返回支付记录数据| D[客户端]
```

### 支付回调处理
处理支付异步回调通知并更新支付状态，定义支付后返回的基础数据对象。

```mermaid
graph TD
    A[支付异步回调通知] --> B[PayNoticeController]
    B --> C[处理支付通知]
    C --> D[更新支付状态]
    D --> E[PayInfoResultBO]
    E --> F[返回支付结果数据]
```

### 支付记录DTO
定义订单支付记录的数据传输对象，包含订单号和支付回跳地址。

```mermaid
graph TD
    A[订单支付请求] --> B[创建PayInfoDTO]
    B --> C[设置订单号]
    B --> D[设置支付回跳地址]
    C --> E[保存支付记录]
    D --> E
    E --> F[返回支付结果]
```

