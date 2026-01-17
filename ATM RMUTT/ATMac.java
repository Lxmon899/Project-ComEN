// ไฟล์: ATMac.java
import java.util.Scanner;

class ATMac extends Person {
    private String accountName;
    private String accountId;
    private String password;
    private int balance;

    public ATMac(String personName, String personId, Gender gender, String accountName, String accountId, String password, int initialBalance) {
        super(personName, personId, gender);
        this.accountName = accountName;
        this.accountId = accountId;
        this.password = password;
        this.balance = initialBalance;
    }

    // Constructor เสริมสำหรับ Admin (ถ้าต้องการ)
    public ATMac(String accountName, String accountId, String password, int balance) {
    super(accountName, accountId, Gender.MALE); // เรียกใช้ Constructor ของ Person (Name, ID, Gender)
    this.accountName = accountName;
    this.accountId = accountId;
    this.password = password;
    this.balance = balance;
    }

    public String getAcName() { return this.accountName; }
    public String getAcId() { return this.accountId; }
    public int getBalance() { return this.balance; }

    public void setAccountName(String accountName) {
        this.accountName = accountName;
    }

    public boolean authenticate(String inputPassword) {
        // ตรวจสอบรหัสผ่าน
        return this.password.equals(inputPassword);
    }
    
    public void deposit(int amount) {
        if (amount > 0) {
            this.balance += amount; 
        }
    }

    public boolean withdraw(int amount, int btcRate, Scanner scanner) {
        System.out.println("Withdraw BTC or Bath ?");
        String choice = scanner.nextLine().trim();

        if (choice.equalsIgnoreCase("BTC")) {
            amount = amount * btcRate; 
        }

        if (amount > 0 && amount <= this.balance) {
            this.balance -= amount; 
            return true;  
        } else {
            System.out.println("Transaction failed: Invalid amount or insufficient balance.");
            return false; 
        }
    }

    public boolean transfer(ATMac targetAccount, int amount, int btcRate, Scanner scanner) {
        if (targetAccount == null || this.accountId.equals(targetAccount.getAcId())) return false;
        if (this.withdraw(amount, btcRate, scanner)) { // ส่ง scanner ต่อไป
            targetAccount.deposit(amount);
            return true; 
        }
        return false;
    }

    public void ChangeCurrency(int rate){
        double btc = (double)this.balance / rate;
        System.out.printf("Balance in BTC: %.6f BTC\n", btc);
    }
    
    public void displayInfo() {
        System.out.println("Account Name: " + this.accountName);
        System.out.println("Account ID: " + this.accountId);
        System.out.println("Balance: " + this.balance);
    }
}