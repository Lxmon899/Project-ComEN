import java.util.Scanner;

public class MainApp {
    private static int btcrate;
    private static final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        ATMManager manager = new ATMManager();

        insertExampleData(manager); // เพิ่มข้อมูลตัวอย่าง

        setupManager(manager);
        
        System.out.print("please enter BTC rate: ");
        btcrate = Integer.parseInt(scanner.nextLine().trim());

        while (true) {
            ATMac loggedInAccount = runLogin(manager);

            if (loggedInAccount != null) {
                System.out.println("\n Login successful: " + loggedInAccount.getAcName());
                showMenu(loggedInAccount, manager);
            } else {
                System.out.println("\n Authentication failed. Invalid Account ID or Password.");
            }
        }
    }

    public static void setupAccounts(ATMManager manager) {
        System.out.print("Step 1: Enter amount of all account: ");
        int acamount = Integer.parseInt(scanner.nextLine().trim());
        System.out.println("Step 2: Enter Detail of each account.");
        for (int i = 0; i < acamount; i++) {
            System.out.println("Person name " + (i + 1) + ": ");
            String personName = scanner.nextLine().trim();
            System.out.println("Person ID " + (i + 1) + ": ");
            String personId = scanner.nextLine().trim();
            System.out.print("Gender (M/F): ");
            String gInput = scanner.nextLine().trim().toUpperCase();
            Person.Gender gender = gInput.startsWith("F") ? Person.Gender.FEMALE : Person.Gender.MALE;
            
            System.out.println("Account " + (i + 1) + " Detail:");
            System.out.print("Account ID: ");
            String acid = scanner.nextLine().trim();
            System.out.print("Account NAME: ");
            String acname = scanner.nextLine().trim();
            System.out.print("PASSWORD: ");
            String acpass = scanner.nextLine().trim();
            System.out.print("Balance: ");
            int acbalance = Integer.parseInt(scanner.nextLine().trim());

            manager.addAccount(new ATMac(personName, personId, acname, acid, acpass, acbalance, gender));
        }
    }

    public static void insertExampleData(ATMManager manager) {
        // ใช้ลำดับ: personName, personId, gender, accountName, accountId, password, balance
        manager.addAccount(new ATMac("Admin_Acc", "admin01", "1234", 0));
        manager.addAccount(new ATMac("Somchai", "P-01", Person.Gender.MALE, "Somchai_Acc", "user01", "5555", 100000));
        manager.addAccount(new ATMac("Somsri", "P-02", Person.Gender.FEMALE, "Somsri_Acc", "user02", "8888", 50000));
        System.out.println("✅ Sample data loaded.");
    }

    public static void setupManager(ATMManager manager) {
        System.out.print("Step 1: Enter amount of all admin: ");
        int adminAmount = Integer.parseInt(scanner.nextLine().trim());

        for (int i = 0; i < adminAmount; i++) {
            System.out.println("Admin " + (i + 1) + " Detail:");
            System.out.print("Admin ID: ");
            String adminId = scanner.nextLine().trim();
            System.out.print("Admin NAME: ");
            String adminName = scanner.nextLine().trim();
            System.out.print("PASSWORD: ");
            String adminPass = scanner.nextLine().trim();

            ATMac adminAcc = new ATMac(adminName, adminId, adminPass, 1312000);
            manager.addAccount(adminAcc);
        }
    }

    public static ATMac runLogin(ATMManager manager) {
        System.out.println("\n--- ATM ComputerThanyaburi Bank Login ---");
        System.out.print("Account ID: ");
        String id = scanner.nextLine().trim();
        System.out.print("PASSWORD: ");
        String pass = scanner.nextLine().trim();
        return manager.login(id, pass);
    }

    public static void showMenu(ATMac account, ATMManager manager) {
        int option;
        boolean isAdmin = account.getAcId().toLowerCase().startsWith("admin");

        do {
            System.out.println("\n--- Welcome, " + account.getAcName() + " ---");
            System.out.println("1. Account Balance | 2. Withdraw | 3. Deposit");
            System.out.println("4. Transfer        | 5. ChangeCurrency");
            if (isAdmin) {
                System.out.println("6. [Admin] List All | 7. [Admin] Edit Name | 8. [Admin] Create Acc");
            }
            System.out.println("9. Exit");
            System.out.print("Choose an option: ");

            try {
                option = Integer.parseInt(scanner.nextLine().trim());
            } catch (NumberFormatException e) {
                option = 0;
            }

            switch (option) {
                case 1:
                    System.out.println("Current Balance: " + account.getBalance());
                    break;
                case 2:
                    System.out.print("Enter amount: ");
                    int wAmt = Integer.parseInt(scanner.nextLine().trim());

                    if (account.withdraw(wAmt, btcrate, scanner)) {
                        System.out.println("Withdrew: " + wAmt);
                    }
                    break;
                case 3:
                    System.out.print("Enter amount: ");
                    int dAmt = Integer.parseInt(scanner.nextLine().trim());
                    account.deposit(dAmt);
                    System.out.println("Deposit Successful!");
                    break;
                case 4:
                    System.out.print("Target Account ID: ");
                    String tId = scanner.nextLine().trim();
                    System.out.print("Amount: ");
                    int tAmt = Integer.parseInt(scanner.nextLine().trim());
                    ATMac target = manager.findAccountById(tId);
                    // ส่ง scanner เข้าไปใน transfer ด้วย
                    if (account.transfer(target, tAmt, btcrate, scanner)) {
                        System.out.println("Transferred Successfully!");
                    } else {
                        System.out.println("Transfer failed.");
                    }
                    break;
                case 5:
                    account.ChangeCurrency(btcrate);
                    break;
                case 6:
                    if (isAdmin) manager.listAllAccounts();
                    break;
                case 7:
                    if (isAdmin) {
                        System.out.print("ID to edit: ");
                        String eid = scanner.nextLine().trim();
                        System.out.print("New Name: ");
                        String ename = scanner.nextLine().trim();
                        manager.updateAccountName(eid, ename);
                    }
                    break;
                case 8:
                    if (isAdmin) setupAccounts(manager);
                    break;
                case 9:
                    System.out.println("Logging out...");
                    break;
                default:
                    System.out.println("Invalid option.");
                    break;
            }
        } while (option != 9);
    }
}