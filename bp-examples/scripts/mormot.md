You can follow the mormot example in the tags <mormot></mormot>:
<mormot>
# Building Web Applications with mORMot2: ORM, SOA, and MVC Guide

## Introduction

This guide teaches developers and AI agents how to create full-stack web applications using the mORMot2 framework. mORMot2 is an open-source toolkit for Delphi and FreePascal that provides ORM (Object-Relational Mapping), SOA (Service-Oriented Architecture), and MVC (Model-View-Controller) capabilities with a web-based client interface.

## Prerequisites

**Supported Compilers:**
- FPC 3.2.3 or later
- Delphi 7 through Delphi 12.2 Athenes

**Installation:**
1. Clone the repository: `git clone https://github.com/synopse/mORMot2.git`
2. Download static libraries from https://synopse.info/files/mormot2static.tgz
3. Extract static files to `mORMot2/static` directory

**IDE Setup (Delphi):**
1. Create environment variable `mormot2` pointing to the `src` subfolder
2. Add to library path: `$(mormot2);$(mormot2)\core;$(mormot2)\lib;$(mormot2)\crypt;$(mormot2)\net;$(mormot2)\db;$(mormot2)\rest;$(mormot2)\orm;$(mormot2)\soa;$(mormot2)\app;$(mormot2)\script;$(mormot2)\ui;$(mormot2)\tools;$(mormot2)\misc`

**IDE Setup (Lazarus):**
1. Open and compile `/packages/lazarus/mormot2.lpk`
2. Optionally compile `mormot2ui.lpk` if UI components needed

## Architecture Overview

A typical mORMot2 web application consists of:

1. **Model (ORM Layer)**: Data entities persisted to databases
2. **SOA Layer**: Business logic exposed as REST services via interfaces
3. **MVC Layer**: Web presentation serving HTML/JavaScript clients
4. **Client**: Web browser consuming REST APIs and rendering UI

## Part 1: Setting Up the Model (ORM)

### Defining Your Data Model

The ORM layer uses classes descended from `TOrm` (formerly `TSQLRecord` in mORMot 1.18) to represent database entities.

```pascal
uses
  mormot.core.base,
  mormot.core.data,
  mormot.orm.core;

type
  TCustomer = class(TOrm)
  private
    fName: RawUtf8;
    fEmail: RawUtf8;
    fCreatedAt: TDateTime;
  published
    property Name: RawUtf8 read fName write fName;
    property Email: RawUtf8 read fEmail write fEmail;
    property CreatedAt: TDateTime read fCreatedAt write fCreatedAt;
  end;

  TOrder = class(TOrm)
  private
    fCustomer: TRecordReference;
    fOrderDate: TDateTime;
    fTotalAmount: Currency;
    fStatus: RawUtf8;
  published
    property Customer: TRecordReference read fCustomer write fCustomer;
    property OrderDate: TDateTime read fOrderDate write fOrderDate;
    property TotalAmount: Currency read fTotalAmount write fTotalAmount;
    property Status: RawUtf8 read fStatus write fStatus;
  end;
```

**Key Points:**
- All ORM classes inherit from `TOrm`
- Use `RawUtf8` for strings (optimized UTF-8 storage)
- `TRecordReference` creates foreign key relationships
- Only `published` properties are persisted
- Each `TOrm` automatically gets an `ID` field

### Creating the ORM Model

```pascal
uses
  mormot.orm.base,
  mormot.rest.sqlite3;

function CreateModel: TOrmModel;
begin
  Result := TOrmModel.Create([TCustomer, TOrder], 'myapp');
end;
```

### Connecting to a Database

**SQLite Example (embedded, zero-config):**

```pascal
uses
  mormot.rest.server,
  mormot.rest.sqlite3;

var
  Model: TOrmModel;
  Server: TRestServerDB;
begin
  Model := CreateModel;
  try
    Server := TRestServerDB.Create(Model, 'data.db3');
    try
      Server.CreateMissingTables; // Auto-create tables if needed
      
      // Server is ready to use
      
    finally
      Server.Free;
    end;
  finally
    Model.Free;
  end;
end;
```

**PostgreSQL Example:**

```pascal
uses
  mormot.db.sql,
  mormot.db.sql.postgres;

var
  Props: TOrmConnectionProperties;
begin
  Props := TOrmConnectionProperties.Create(
    TSqlDBPostgresConnectionProperties,
    'Server=localhost;Port=5432;Database=mydb;User=postgres;Password=secret',
    'mydb',
    '',
    ''
  );
  
  Server := TRestServerDB.Create(Model, Props);
  Server.CreateMissingTables;
end;
```

### Basic ORM Operations

```pascal
// CREATE - Adding a new record
var
  Customer: TCustomer;
begin
  Customer := TCustomer.Create;
  try
    Customer.Name := 'John Doe';
    Customer.Email := 'john@example.com';
    Customer.CreatedAt := NowUtc;
    Server.Orm.Add(Customer, true); // Second param: send response
  finally
    Customer.Free;
  end;
end;

// READ - Retrieving records
var
  Customer: TCustomer;
begin
  Customer := TCustomer.Create(Server.Orm, 1); // Load by ID
  try
    WriteLn('Customer: ', Customer.Name);
  finally
    Customer.Free;
  end;
end;

// UPDATE - Modifying records
begin
  Customer.Email := 'newemail@example.com';
  Server.Orm.Update(Customer);
end;

// DELETE - Removing records
begin
  Server.Orm.Delete(TCustomer, 1); // Delete by ID
end;

// QUERY - Advanced retrieval
var
  Customers: TObjectList<TCustomer>;
begin
  Customers := Server.Orm.RetrieveList<TCustomer>(
    'Name LIKE ?', ['%John%']
  );
  try
    for Customer in Customers do
      WriteLn(Customer.Name);
  finally
    Customers.Free;
  end;
end;
```

## Part 2: Building the SOA Layer

### Defining Service Interfaces

Business logic is exposed through interface-based services that mORMot2 automatically translates to REST endpoints.

```pascal
uses
  mormot.core.interfaces;

type
  ICustomerService = interface(IInvokable)
    ['{A1234567-89AB-CDEF-0123-456789ABCDEF}']
    
    function GetCustomerInfo(CustomerID: TID): Variant;
    function CreateCustomer(const Name, Email: RawUtf8): TID;
    function SearchCustomers(const SearchTerm: RawUtf8): TVariantDynArray;
    procedure UpdateCustomerEmail(CustomerID: TID; const NewEmail: RawUtf8);
    function GetCustomerOrders(CustomerID: TID): TVariantDynArray;
  end;
```

**Key Points:**
- Interfaces must inherit from `IInvokable`
- Each interface needs a unique GUID
- Methods become REST endpoints automatically
- Use simple types (integers, strings, Variant) for parameters
- Return types can be simple values, Variant, or dynamic arrays

### Implementing Services

```pascal
type
  TCustomerService = class(TInjectableObjectRest, ICustomerService)
  public
    function GetCustomerInfo(CustomerID: TID): Variant;
    function CreateCustomer(const Name, Email: RawUtf8): TID;
    function SearchCustomers(const SearchTerm: RawUtf8): TVariantDynArray;
    procedure UpdateCustomerEmail(CustomerID: TID; const NewEmail: RawUtf8);
    function GetCustomerOrders(CustomerID: TID): TVariantDynArray;
  end;

function TCustomerService.GetCustomerInfo(CustomerID: TID): Variant;
var
  Customer: TCustomer;
begin
  Customer := TCustomer.Create(Server.Orm, CustomerID);
  try
    if Customer.ID = 0 then
      raise EServiceException.Create('Customer not found');
    
    TDocVariantData(Result).InitObject([
      'id', Customer.ID,
      'name', Customer.Name,
      'email', Customer.Email,
      'createdAt', DateTimeToIso8601(Customer.CreatedAt, true)
    ]);
  finally
    Customer.Free;
  end;
end;

function TCustomerService.CreateCustomer(const Name, Email: RawUtf8): TID;
var
  Customer: TCustomer;
begin
  Customer := TCustomer.Create;
  try
    Customer.Name := Name;
    Customer.Email := Email;
    Customer.CreatedAt := NowUtc;
    Result := Server.Orm.Add(Customer, true);
  finally
    Customer.Free;
  end;
end;

function TCustomerService.SearchCustomers(const SearchTerm: RawUtf8): TVariantDynArray;
var
  Customers: TObjectList<TCustomer>;
  Customer: TCustomer;
  i: integer;
begin
  Customers := Server.Orm.RetrieveList<TCustomer>(
    'Name LIKE ? OR Email LIKE ?', 
    ['%' + SearchTerm + '%', '%' + SearchTerm + '%']
  );
  try
    SetLength(Result, Customers.Count);
    for i := 0 to Customers.Count - 1 do
    begin
      Customer := Customers[i];
      TDocVariantData(Result[i]).InitObject([
        'id', Customer.ID,
        'name', Customer.Name,
        'email', Customer.Email
      ]);
    end;
  finally
    Customers.Free;
  end;
end;

procedure TCustomerService.UpdateCustomerEmail(CustomerID: TID; const NewEmail: RawUtf8);
var
  Customer: TCustomer;
begin
  Customer := TCustomer.Create(Server.Orm, CustomerID);
  try
    if Customer.ID = 0 then
      raise EServiceException.Create('Customer not found');
    
    Customer.Email := NewEmail;
    Server.Orm.Update(Customer);
  finally
    Customer.Free;
  end;
end;

function TCustomerService.GetCustomerOrders(CustomerID: TID): TVariantDynArray;
var
  Orders: TObjectList<TOrder>;
  Order: TOrder;
  i: integer;
begin
  Orders := Server.Orm.RetrieveList<TOrder>('Customer=?', [CustomerID]);
  try
    SetLength(Result, Orders.Count);
    for i := 0 to Orders.Count - 1 do
    begin
      Order := Orders[i];
      TDocVariantData(Result[i]).InitObject([
        'id', Order.ID,
        'orderDate', DateTimeToIso8601(Order.OrderDate, true),
        'totalAmount', Order.TotalAmount,
        'status', Order.Status
      ]);
    end;
  finally
    Orders.Free;
  end;
end;
```

### Registering Services

```pascal
uses
  mormot.rest.http.server;

begin
  // Register the service implementation
  Server.ServiceDefine(TCustomerService, [ICustomerService], sicShared);
  
  // sicShared: one instance shared across all requests (stateless)
  // sicSingle: one instance per session (stateful)
  // sicClientDriven: instance lifecycle managed by client
  // sicPerSession: one instance per authenticated session
  // sicPerUser: one instance per authenticated user
  // sicPerGroup: one instance per user group
  // sicPerThread: one instance per thread
end;
```

## Part 3: Creating the HTTP Server with MVC

### Setting Up the HTTP Server

```pascal
uses
  mormot.rest.http.server,
  mormot.net.http,
  mormot.net.server;

var
  HttpServer: TRestHttpServer;
begin
  // Create HTTP server on port 8080
  HttpServer := TRestHttpServer.Create(
    '8080',           // Port
    [Server],         // REST server instances
    '+',              // Domain binding (+ = all)
    useHttpAsync      // Server model (async for performance)
  );
  
  // Configure CORS if needed for web clients
  HttpServer.AccessControlAllowOrigin := '*';
  
  WriteLn('Server running on http://localhost:8080');
  WriteLn('Press [Enter] to quit');
  ReadLn;
  
  HttpServer.Free;
end;
```

**Server Models:**
- `useHttpSocket`: Simple blocking sockets (legacy)
- `useHttpAsync`: High-performance async I/O (recommended)
- `useHttpApiRegisteringURI`: Windows HTTP.sys integration

### Creating MVC Views

mORMot2 provides Mustache templating for server-side rendering:

```pascal
uses
  mormot.core.mustache;

type
  TMyMvcApplication = class(TMvcApplication)
  published
    procedure Customers(var Scope: variant);
    procedure CustomerDetail(CustomerID: TID; var Scope: variant);
  end;

procedure TMyMvcApplication.Customers(var Scope: variant);
var
  Customers: TObjectList<TCustomer>;
  CustomerArray: TVariantDynArray;
  i: integer;
begin
  // Fetch data
  Customers := Server.Orm.RetrieveList<TCustomer>('', []);
  try
    // Convert to variant array for template
    SetLength(CustomerArray, Customers.Count);
    for i := 0 to Customers.Count - 1 do
      TDocVariantData(CustomerArray[i]).InitObject([
        'id', Customers[i].ID,
        'name', Customers[i].Name,
        'email', Customers[i].Email
      ]);
    
    // Pass to view
    TDocVariantData(Scope).InitObject([
      'title', 'Customer List',
      'customers', Variant(CustomerArray)
    ]);
  finally
    Customers.Free;
  end;
end;

procedure TMyMvcApplication.CustomerDetail(CustomerID: TID; var Scope: variant);
var
  Customer: TCustomer;
  Service: ICustomerService;
  Orders: TVariantDynArray;
begin
  Customer := TCustomer.Create(Server.Orm, CustomerID);
  try
    if Customer.ID = 0 then
      raise EMvcApplication.CreateGotoError(HTTP_NOTFOUND, 'Customer not found');
    
    // Get orders via service
    if Server.Services.Resolve(ICustomerService, Service) then
      Orders := Service.GetCustomerOrders(CustomerID);
    
    TDocVariantData(Scope).InitObject([
      'title', 'Customer: ' + Customer.Name,
      'customer', _ObjFast([
        'id', Customer.ID,
        'name', Customer.Name,
        'email', Customer.Email
      ]),
      'orders', Variant(Orders)
    ]);
  finally
    Customer.Free;
  end;
end;
```

**Mustache Template Example** (`customers.html`):

```html
<!DOCTYPE html>
<html>
<head>
    <title>{{title}}</title>
    <style>
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
    </style>
</head>
<body>
    <h1>{{title}}</h1>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Email</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            {{#customers}}
            <tr>
                <td>{{id}}</td>
                <td>{{name}}</td>
                <td>{{email}}</td>
                <td><a href="/customer/{{id}}">View Details</a></td>
            </tr>
            {{/customers}}
        </tbody>
    </table>
</body>
</html>
```

### Registering MVC Application

```pascal
var
  MvcApp: TMyMvcApplication;
begin
  MvcApp := TMyMvcApplication.Create(Server);
  Server.RootRedirectToUri('customers'); // Default page
end;
```

## Part 4: Building the Web Client

### REST API Client (JavaScript)

Create a modern web application consuming the SOA services:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Customer Management</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .customer-card { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        input { padding: 8px; margin: 5px; width: 200px; }
    </style>
</head>
<body>
    <h1>Customer Management</h1>
    
    <div id="create-section">
        <h2>Create New Customer</h2>
        <input type="text" id="newName" placeholder="Name">
        <input type="email" id="newEmail" placeholder="Email">
        <button onclick="createCustomer()">Create</button>
    </div>
    
    <div id="search-section">
        <h2>Search Customers</h2>
        <input type="text" id="searchTerm" placeholder="Search...">
        <button onclick="searchCustomers()">Search</button>
    </div>
    
    <div id="results"></div>

    <script>
        const API_BASE = 'http://localhost:8080';
        
        async function callService(interfaceName, methodName, params = []) {
            const url = `${API_BASE}/${interfaceName}/${methodName}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        }
        
        async function createCustomer() {
            const name = document.getElementById('newName').value;
            const email = document.getElementById('newEmail').value;
            
            try {
                const customerId = await callService('CustomerService', 'CreateCustomer', [name, email]);
                alert(`Customer created with ID: ${customerId}`);
                document.getElementById('newName').value = '';
                document.getElementById('newEmail').value = '';
            } catch (error) {
                alert('Error creating customer: ' + error.message);
            }
        }
        
        async function searchCustomers() {
            const searchTerm = document.getElementById('searchTerm').value;
            
            try {
                const customers = await callService('CustomerService', 'SearchCustomers', [searchTerm]);
                displayCustomers(customers);
            } catch (error) {
                alert('Error searching customers: ' + error.message);
            }
        }
        
        async function loadCustomerDetails(customerId) {
            try {
                const info = await callService('CustomerService', 'GetCustomerInfo', [customerId]);
                const orders = await callService('CustomerService', 'GetCustomerOrders', [customerId]);
                
                let html = `
                    <div class="customer-card">
                        <h3>${info.name}</h3>
                        <p>Email: ${info.email}</p>
                        <p>Created: ${info.createdAt}</p>
                        <h4>Orders (${orders.length})</h4>
                        <ul>
                `;
                
                orders.forEach(order => {
                    html += `<li>Order #${order.id}: ${order.status} - $${order.totalAmount}</li>`;
                });
                
                html += '</ul></div>';
                document.getElementById('results').innerHTML = html;
            } catch (error) {
                alert('Error loading customer: ' + error.message);
            }
        }
        
        function displayCustomers(customers) {
            let html = '<h2>Search Results</h2>';
            
            customers.forEach(customer => {
                html += `
                    <div class="customer-card">
                        <h3>${customer.name}</h3>
                        <p>Email: ${customer.email}</p>
                        <button onclick="loadCustomerDetails(${customer.id})">View Details</button>
                    </div>
                `;
            });
            
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
```

### RESTful URL Patterns

mORMot2 automatically exposes these REST endpoints:

**ORM REST Endpoints:**
- `GET /Customer/123` - Retrieve customer by ID
- `GET /Customer?where=Name%3D%27John%27` - Query customers
- `POST /Customer` - Create new customer (JSON body)
- `PUT /Customer` - Update customer (JSON body)
- `DELETE /Customer/123` - Delete customer by ID

**SOA Service Endpoints:**
- `POST /CustomerService/GetCustomerInfo` - Body: `[123]`
- `POST /CustomerService/SearchCustomers` - Body: `["john"]`
- Method-based routing with JSON-RPC style invocation

## Part 5: Complete Application Example

```pascal
program WebShopServer;

{$APPTYPE CONSOLE}

uses
  SysUtils,
  mormot.core.base,
  mormot.core.data,
  mormot.core.json,
  mormot.core.interfaces,
  mormot.core.mustache,
  mormot.orm.core,
  mormot.rest.core,
  mormot.rest.server,
  mormot.rest.sqlite3,
  mormot.rest.http.server,
  mormot.net.http,
  mormot.net.server;

type
  // Models
  TCustomer = class(TOrm)
  private
    fName: RawUtf8;
    fEmail: RawUtf8;
  published
    property Name: RawUtf8 read fName write fName;
    property Email: RawUtf8 read fEmail write fEmail;
  end;

  // Service Interface
  ICustomerService = interface(IInvokable)
    ['{12345678-1234-1234-1234-123456789012}']
    function CreateCustomer(const Name, Email: RawUtf8): TID;
    function SearchCustomers(const Term: RawUtf8): TVariantDynArray;
  end;

  // Service Implementation
  TCustomerService = class(TInjectableObjectRest, ICustomerService)
  public
    function CreateCustomer(const Name, Email: RawUtf8): TID;
    function SearchCustomers(const Term: RawUtf8): TVariantDynArray;
  end;

function TCustomerService.CreateCustomer(const Name, Email: RawUtf8): TID;
var
  Customer: TCustomer;
begin
  Customer := TCustomer.Create;
  try
    Customer.Name := Name;
    Customer.Email := Email;
    Result := Server.Orm.Add(Customer, true);
  finally
    Customer.Free;
  end;
end;

function TCustomerService.SearchCustomers(const Term: RawUtf8): TVariantDynArray;
var
  Customers: TObjectList<TCustomer>;
  i: integer;
begin
  Customers := Server.Orm.RetrieveList<TCustomer>(
    'Name LIKE ? OR Email LIKE ?', ['%' + Term + '%', '%' + Term + '%']);
  try
    SetLength(Result, Customers.Count);
    for i := 0 to Customers.Count - 1 do
      TDocVariantData(Result[i]).InitObject([
        'id', Customers[i].ID,
        'name', Customers[i].Name,
        'email', Customers[i].Email
      ]);
  finally
    Customers.Free;
  end;
end;

var
  Model: TOrmModel;
  Server: TRestServerDB;
  HttpServer: TRestHttpServer;
begin
  // Create model
  Model := TOrmModel.Create([TCustomer], 'webshop');
  try
    // Create REST server with SQLite database
    Server := TRestServerDB.Create(Model, 'data.db3');
    try
      Server.CreateMissingTables;
      
      // Register SOA service
      Server.ServiceDefine(TCustomerService, [ICustomerService], sicShared);
      
      // Create HTTP server
      HttpServer := TRestHttpServer.Create('8080', [Server], '+', useHttpAsync);
      try
        HttpServer.AccessControlAllowOrigin := '*';
        
        WriteLn('WebShop Server running on http://localhost:8080');
        WriteLn('REST ORM: http://localhost:8080/Customer');
        WriteLn('SOA Services: http://localhost:8080/CustomerService');
        WriteLn('Press [Enter] to quit');
        ReadLn;
      finally
        HttpServer.Free;
      end;
    finally
      Server.Free;
    end;
  finally
    Model.Free;
  end;
end.
```

## Advanced Topics

### Authentication and Security

```pascal
// Add user authentication
Server.HandleAuthentication := true;

// Create admin user
Server.Orm.Add(TAuthUser.Create('admin', 'password'), true);

// Secure service methods
type
  ISecureService = interface(IInvokable)
    ['{...}']
    ['{auth:true}'] // Require authentication
    function SecureMethod: RawUtf8;
  end;
```

### WebSockets Support

```pascal
// Enable WebSockets for real-time communication
HttpServer.WebSocketsEnable(Server, 'myapp');

// Client can connect via: ws://localhost:8080/myapp
```

### HTTPS with Let's Encrypt

```pascal
uses
  mormot.net.acme;

// Automatic HTTPS with Let's Encrypt
HttpServer.UseSSL('yourdomain.com', 'admin@yourdomain.com');
```

### OpenAPI/Swagger Documentation

```pascal
// Automatic API documentation
Server.ServiceMethodRegisterPublishedMethods('', sicShared);
// Access at: http://localhost:8080/api-docs
```

## Best Practices

1. **Separation of Concerns**: Keep ORM models, services, and views in separate units
2. **Use Interfaces**: Always define services via interfaces for flexibility
3. **Stateless Services**: Prefer `sicShared` for scalability
4. **Error Handling**: Use `EServiceException` for business logic errors
5. **UTF-8 Everywhere**: Use `RawUtf8` for all string operations
6. **Async Operations**: Use `useHttpAsync` for production servers
7. **Database Pooling**: For external databases, configure connection pools appropriately
8. **Logging**: Enable framework logging for debugging

## Troubleshooting

**Common Issues:**

1. **Service not found**: Ensure interface has GUID and is registered with `ServiceDefine`
2. **CORS errors**: Set `HttpServer.AccessControlAllowOrigin` for web clients
3. **Database locked**: Use WAL mode for SQLite: `Server.DB.ExecuteNoResult('PRAGMA journal_mode=WAL')`
4. **Memory leaks**: Always free TOrm instances in try/finally blocks

## Resources

- Official Documentation: Work in progress at synopse.info
- Examples: See `/ex` folder in repository
- Thomas Tutorials: Comprehensive learning materials in examples
- Forum: https://synopse.info (official support)
- Telegram: mORMot Telegram group
- Discord: mORMot Discord server

## Migration from mORMot 1.18

Key changes when upgrading:
- `TSQLRecord` → `TOrm`
- `TSQLRest` → `TRest`
- Units reorganized: use specific domain units (orm, rest, soa, app)
- Enhanced RTTI and JSON performance
- New async server models available

## Conclusion

mORMot2 provides a complete, high-performance framework for building modern web applications with Delphi and FreePascal. Its integrated ORM, SOA, and MVC capabilities allow developers to create scalable client-server applications with minimal boilerplate code. The framework's convention-over-configuration approach and automatic REST endpoint generation significantly reduce development time while maintaining flexibility and performance.
</mormot>