public class Person {
    protected String personName;
    protected String personId;
    protected Gender gender;
    public enum Gender {
        MALE, FEMALE
    }

    public Person(String personName, String personId, Gender gender) {
        this.personName = personName;
        this.personId = personId;
        this.gender = gender;
    }
}