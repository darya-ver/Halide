#ifndef HALIDE_ERROR_H
#define HALIDE_ERROR_H

#include <sstream>
#include <stdexcept>

#include "Debug.h"
#include "runtime/HalideRuntime.h"  // for HALIDE_ALWAYS_INLINE

namespace Halide {

/** Query whether Halide was compiled with exceptions. */
bool exceptions_enabled();

/** A base class for Halide errors.
 *
 * Note that this deliberately does *not* descend from std::runtime_error, or
 * even std::exception; unfortunately, std::runtime_error is not marked as
 * DLLEXPORT on Windows, but Error needs to be marked as such, and mismatching
 * DLLEXPORT annotations in a class inheritance hierarchy in this way can lead
 * to ODR violations. Instead, we just attempt to replicate the API of
 * runtime_error here. */
struct HALIDE_EXPORT_SYMBOL Error {
    Error() = delete;

    // Give each class a non-inlined constructor so that the type
    // doesn't get separately instantiated in each compilation unit.
    explicit Error(const char *msg);
    explicit Error(const std::string &msg);

    Error(const Error &);
    Error &operator=(const Error &);
    Error(Error &&) noexcept;
    Error &operator=(Error &&) noexcept;

    virtual ~Error();

    virtual const char *what() const noexcept;

private:
    // Using a std::string here will cause MSVC to complain about the fact
    // that class std::string isn't declared DLLEXPORT, even though the
    // field is private; rather than suppress the warning, we'll just use
    // an old-fashioned new-and-delete to keep it nice and clean.
    char *what_;
};

/** An error that occurs while running a JIT-compiled Halide pipeline. */
struct HALIDE_EXPORT_SYMBOL RuntimeError : public Error {
    explicit RuntimeError(const char *msg);
    explicit RuntimeError(const std::string &msg);
};

/** An error that occurs while compiling a Halide pipeline that Halide
 * attributes to a user error. */
struct HALIDE_EXPORT_SYMBOL CompileError : public Error {
    explicit CompileError(const char *msg);
    explicit CompileError(const std::string &msg);
};

/** An error that occurs while compiling a Halide pipeline that Halide
 * attributes to an internal compiler bug, or to an invalid use of
 * Halide's internals. */
struct HALIDE_EXPORT_SYMBOL InternalError : public Error {
    explicit InternalError(const char *msg);
    explicit InternalError(const std::string &msg);
};

/** CompileTimeErrorReporter is used at compile time (*not* runtime) when
 * an error or warning is generated by Halide. Note that error() is called
 * a fatal error has occurred, and returning to Halide may cause a crash;
 * implementations of CompileTimeErrorReporter::error() should never return.
 * (Implementations of CompileTimeErrorReporter::warning() may return but
 * may also abort(), exit(), etc.)
 */
class CompileTimeErrorReporter {
public:
    virtual ~CompileTimeErrorReporter() = default;
    virtual void warning(const char *msg) = 0;
    virtual void error(const char *msg) = 0;
};

/** The default error reporter logs to stderr, then throws an exception
 * (if HALIDE_WITH_EXCEPTIONS) or calls abort (if not). This allows customization
 * of that behavior if a more gentle response to error reporting is desired.
 * Note that error_reporter is expected to remain valid across all Halide usage;
 * it is up to the caller to ensure that this is the case (and to do any
 * cleanup necessary).
 */
void set_custom_compile_time_error_reporter(CompileTimeErrorReporter *error_reporter);

namespace Internal {

struct ErrorReport {
    enum {
        User = 0x0001,
        Warning = 0x0002,
        Runtime = 0x0004
    };

    std::ostringstream msg;
    const int flags;

    ErrorReport(const char *f, int l, const char *cs, int flags);

    // Just a trick used to convert RValue into LValue
    HALIDE_ALWAYS_INLINE ErrorReport &ref() {
        return *this;
    }

    template<typename T>
    ErrorReport &operator<<(const T &x) {
        msg << x;
        return *this;
    }

    /** When you're done using << on the object, and let it fall out of
     * scope, this errors out, or throws an exception if they are
     * enabled. This is a little dangerous because the destructor will
     * also be called if there's an exception in flight due to an
     * error in one of the arguments passed to operator<<. We handle
     * this by only actually throwing if there isn't an exception in
     * flight already.
     */
    ~ErrorReport() noexcept(false);
};

// This uses operator precedence as a trick to avoid argument evaluation if
// an assertion is true: it is intended to be used as part of the
// _halide_internal_assertion macro, to coerce the result of the stream
// expression to void (to match the condition-is-false case).
class Voidifier {
public:
    HALIDE_ALWAYS_INLINE Voidifier() = default;
    // This has to be an operator with a precedence lower than << but
    // higher than ?:
    HALIDE_ALWAYS_INLINE void operator&(ErrorReport &) {
    }
};

/**
 * _halide_internal_assertion is used to implement our assertion macros
 * in such a way that the messages output for the assertion are only
 * evaluated if the assertion's value is false.
 *
 * Note that this macro intentionally has no parens internally; in actual
 * use, the implicit grouping will end up being
 *
 *   condition ? (void) : (Voidifier() & (ErrorReport << arg1 << arg2 ... << argN))
 *
 * This (regrettably) requires a macro to work, but has the highly desirable
 * effect that all assertion parameters are totally skipped (not ever evaluated)
 * when the assertion is true.
 */
#define _halide_internal_assertion(condition, flags) \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    (condition) ? (void)0 : ::Halide::Internal::Voidifier() & ::Halide::Internal::ErrorReport(__FILE__, __LINE__, #condition, flags).ref()

#define internal_error Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, 0)
#define user_error Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, Halide::Internal::ErrorReport::User)
#define user_warning Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, Halide::Internal::ErrorReport::User | Halide::Internal::ErrorReport::Warning)
#define halide_runtime_error Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, Halide::Internal::ErrorReport::User | Halide::Internal::ErrorReport::Runtime)

#define internal_assert(c) _halide_internal_assertion(c, 0)
#define user_assert(c) _halide_internal_assertion(c, Halide::Internal::ErrorReport::User)

// The nicely named versions get cleaned up at the end of Halide.h,
// but user code might want to do halide-style user_asserts (e.g. the
// Extern macros introduce calls to user_assert), so for that purpose
// we define an equivalent macro that can be used outside of Halide.h
#define _halide_user_assert(c) _halide_internal_assertion(c, Halide::Internal::ErrorReport::User)

// N.B. Any function that might throw a user_assert or user_error may
// not be inlined into the user's code, or the line number will be
// misattributed to Halide.h. Either make such functions internal to
// libHalide, or mark them as HALIDE_NO_USER_CODE_INLINE.

}  // namespace Internal

}  // namespace Halide

#endif
