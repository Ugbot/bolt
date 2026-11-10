// Generates the tables of include/bolt/kernels/bolt_unicode_class.h from the running JDK:
//   java tools/GenUnicodeClass.java > tables.inc   (paste after the UniRange struct)
// Character.getType UPPERCASE_LETTER / LOWERCASE_LETTER / DECIMAL_DIGIT_NUMBER over the BMP (what Spark mask() uses), as maximal runs.
public class GenUnicodeClass {
    interface P { boolean t(int cp); }
    static void emit(String name, P p) {
        StringBuilder sb = new StringBuilder();
        int n = 0, start = -1;
        for (int cp = 0; cp <= 0x110000; ++cp) {
            boolean in = cp <= 0x10FFFF && p.t(cp);
            if (in && start < 0) start = cp;
            if (!in && start >= 0) {
                if (n % 6 == 0) sb.append("\n    ");
                sb.append(String.format("{0x%X, 0x%X}, ", start, cp - 1));
                n++; start = -1;
            }
        }
        System.out.println("inline constexpr UniRange " + name + "[" + n + "] = {" + sb + "\n};");
        System.out.println("inline constexpr uint32_t " + name + "Count = " + n + ";");
    }
    public static void main(String[] a) {
        emit("kUpper", cp -> cp <= 0xFFFF && Character.getType(cp) == Character.UPPERCASE_LETTER);
        emit("kLower", cp -> cp <= 0xFFFF && Character.getType(cp) == Character.LOWERCASE_LETTER);
        emit("kDigit", cp -> cp <= 0xFFFF && Character.getType(cp) == Character.DECIMAL_DIGIT_NUMBER);
        System.err.println("unicode " + Character.UnicodeBlock.of('a') + " " + System.getProperty("java.version"));
    }
}
