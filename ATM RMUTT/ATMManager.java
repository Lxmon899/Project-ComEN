import java.util.ArrayList;
import java.util.List;

class ATMManager {
    // เก็บทุกบัญชีไว้ใน List เดียวกัน (ทั้ง Admin และ User)
    private List<ATMac> accounts;

    public ATMManager() {
        this.accounts = new ArrayList<>();
    }

    public void addAccount(ATMac account) {
        this.accounts.add(account);
    }

    public ATMac login(String inputId, String inputPassword) {
        for (ATMac account : this.accounts) {
            // ค้นหา Account ID ที่ตรงกัน
            if (account.getAcId().equals(inputId)) { 
                if (account.authenticate(inputPassword)) {
                    return account;
                }
                return null; 
            }
        }
        return null;
    }
    
    //Method ค้นหาบัญชี
    public ATMac findAccountById(String id) {
        for (ATMac acc : accounts) {
            if (acc.getAcId().equals(id)) return acc;
        }
        return null;
    }

    // Method สำหรับลบบัญชีออกจากระบบ
    public boolean deleteAccount(String id) {
        ATMac account = findAccountById(id);
        if (account != null) {
            accounts.remove(account);
            return true;
        }
        return false;
    }

    public void listAllAccounts() {
    System.out.println("\n--- All Accounts List ---");
    for (ATMac acc : accounts) {
        System.out.println("ID: " + acc.getAcId() + " | Name: " + acc.getAcName() + " | Balance: " + acc.getBalance());
    }
}

    public boolean updateAccountName(String id, String newName) {
        ATMac acc = findAccountById(id);
        if (acc != null) {
            acc.setAccountName(newName);
            return true;
        }
        return false;
    }
}